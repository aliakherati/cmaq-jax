!> Golden harness for HADV -- CMAQ's whole horizontal-advection driver.
!>
!>   usage: harness_hadv <input.bin> <output.bin>
!>
!> Unlike the kernel harnesses, this one runs the complete chain unmodified:
!>
!>   hadvppm.F   layer loop, sub-cycling to the sync step, X-Y/Y-X alternation
!>     x_ppm.F   row gather, west/east boundary condition, sweep
!>     y_ppm.F   column gather, south/north boundary condition, sweep
!>   hcontvel.F  contravariant velocity from the meteorology
!>     zfdbc.f   zero-flux-divergence outflow condition
!>     hppm.F    the PPM kernel itself
!>
!> Only the *sources of data* are replaced: `interpolate_var` reads from a table
!> the harness fills instead of an I/O API file, and RDBCON hands back a
!> preloaded boundary field instead of opening one. That makes the alternation
!> parity and the per-layer sub-stepping -- the two things a property test
!> cannot check -- part of the golden.
!>
!> One call per process, for the usual reason: x_ppm.F, y_ppm.F and hppm.F all
!> allocate on first call and SAVE, sized from NCOLS/NROWS/N_SPC_ADV.
!>
!> Wind convention: with `cstaguv` true (the default, and what MCIP has written
!> since v3.5) hcontvel.F returns UWINDC/VWINDC directly and never touches
!> density -- see hcontvel.F:245-260, which RETURNs early. The UHAT_JD/DENSA_J
!> path is the pre-2009 fallback.
!>
!> `ncalls` drives the alternation. hadvppm.F keeps a SAVEd XYFIRST flag per
!> layer, initially true, and flips it every call so consecutive sync steps
!> sweep X-then-Y and then Y-then-X. A single call therefore only ever exercises
!> one order; asking for two covers both.
!>
!>   input:  ncols, nrows, nlays, ntrns          (4 x int32)
!>           ncalls                              (int32)
!>           jdate, jtime                        (2 x int32)
!>           tstep(3)                            (3 x int32)   HHMMSS
!>           astep(nlays)                        (nlays int32) HHMMSS
!>           xcell, ycell                        (2 x float32) metres
!>           cgrid(ncols, nrows, nlays, ntrns+1) float32
!>           uwindc(ncols+1, nrows, nlays)       float32
!>           vwindc(ncols, nrows+1, nlays)       float32
!>           bcon(nbndy, ntrns+1, nlays)         float32
!>   output: cgrid(ncols, nrows, nlays, ntrns+1) float32, advected in place
program harness_hadv

   use HGRD_DEFN_STUB, only: set_hgrid, NBNDY, XCELL_GD, YCELL_GD, GDTYP_GD
   use VGRD_DEFN, only: NLAYS
   use CGRID_SPCS, only: set_species
   use CENTRALIZED_IO_MODULE, only: cio_init, cio_put
   use BCON_STORE, only: bcon_init, bcon_data
   use UTILIO_DEFN, only: set_file_vars, NEXTIME

   implicit none

   interface
      ! CGRID is a POINTER dummy, so an explicit interface is mandatory.
      subroutine HADV(CGRID, JDATE, JTIME, TSTEP, ASTEP)
         real, pointer :: CGRID(:, :, :, :)
         integer, intent(in) :: JDATE, JTIME
         integer, intent(in) :: TSTEP(3)
         integer, intent(in) :: ASTEP(:)
      end subroutine HADV
   end interface

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios
   integer :: ncols, nrows, nlays_in, ntrns, nspc_adv, ncalls, call_index
   integer :: jdate, jtime
   integer :: tstep(3)
   integer, allocatable :: astep(:)
   real :: xcell, ycell
   real, pointer :: cgrid(:, :, :, :) => null()
   real, allocatable :: uwindc(:, :, :), vwindc(:, :, :)
   real, allocatable :: staged(:, :, :)

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_hadv <input.bin> <output.bin>'
      stop 2
   end if
   call get_command_argument(1, in_path)
   call get_command_argument(2, out_path)

   open (newunit=unit_in, file=trim(in_path), access='stream', &
         form='unformatted', status='old', action='read', iostat=ios)
   if (ios /= 0) then
      write (*, '(a)') 'cannot open input: '//trim(in_path)
      stop 2
   end if

   read (unit_in) ncols, nrows, nlays_in, ntrns
   read (unit_in) ncalls
   read (unit_in) jdate, jtime
   read (unit_in) tstep

   allocate (astep(nlays_in))
   read (unit_in) astep
   read (unit_in) xcell, ycell

   ! --- configure the grid and species layout -----------------------------
   call set_hgrid(ncols, nrows)
   NLAYS = nlays_in
   XCELL_GD = real(xcell, kind(XCELL_GD))
   YCELL_GD = real(ycell, kind(YCELL_GD))
   GDTYP_GD = 2                      ! Lambert: cell widths are already metres
   call set_species(ntrns)
   nspc_adv = ntrns + 1              ! transported species plus rho*J

   ! --- meteorology --------------------------------------------------------
   call cio_init(ncols, nrows, nlays_in)
   ! hcontvel.F looks for UWINDC in the dot-point file to decide whether
   ! C-staggered winds exist; declare both so it takes the modern path. It
   ! blanks VNAME3D just before querying, so this has to go through DESC3
   ! rather than into the common directly.
   call set_file_vars(['UWINDC', 'VWINDC'])

   allocate (uwindc(ncols + 1, nrows, nlays_in))
   allocate (vwindc(ncols, nrows + 1, nlays_in))
   read (unit_in) uwindc
   read (unit_in) vwindc

   ! The field table is dimensioned (ncols+1, nrows+1, nlays); each component
   ! occupies the corner of it that matches its own staggering.
   allocate (staged(ncols + 1, nrows + 1, nlays_in))
   staged = 0.0
   staged(1:ncols + 1, 1:nrows, :) = uwindc
   call cio_put('UWINDC', staged)
   staged = 0.0
   staged(1:ncols, 1:nrows + 1, :) = vwindc
   call cio_put('VWINDC', staged)

   ! --- boundary concentrations -------------------------------------------
   call bcon_init(NBNDY, nspc_adv, nlays_in)
   read (unit_in) bcon_data

   ! --- concentrations -----------------------------------------------------
   allocate (cgrid(ncols, nrows, nlays_in, nspc_adv))
   read (unit_in) cgrid
   close (unit_in)

   do call_index = 1, ncalls
      call HADV(cgrid, jdate, jtime, tstep, astep)
      ! Advance to the next sync step, as sciproc.F does between calls.
      call NEXTIME(jdate, jtime, tstep(2))
   end do

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) cgrid
   close (unit_out)

end program harness_hadv
