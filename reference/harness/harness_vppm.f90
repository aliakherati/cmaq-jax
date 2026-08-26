!> Golden harness for VPPM -- CMAQ's 1-D non-uniform-spacing PPM kernel.
!>
!>   usage: harness_vppm <input.bin> <output.bin>
!>
!> Reads one column, makes exactly ONE call to the unmodified vppm.F, and
!> writes the updated concentrations plus the ADJUSTED face velocities.
!>
!> One call per process, for the same reason as harness_hppm: VPPM computes
!> N_SPC_ADV on its first call and SAVEs it (vppm.F:174-177), and its inner PPM
!> routine precomputes the non-uniform mesh coefficients from DS on ITS first
!> call and SAVEs those too (vppm.F:450-468). Re-calling with a different NI or
!> DS silently reuses the first set.
!>
!> VEL is INTENT(INOUT) and is genuinely modified: vppm.F:200-246 rescales each
!> face velocity so the PPM flux of the rho*J column reproduces the donor-cell
!> flux FLX*dt. That adjusted velocity is then used for every species, so it is
!> part of the golden, not a detail.
!>
!>   input:  ni, nspcs                          (2 x int32)
!>           dt                                 float32
!>           ds(ni)                             float32
!>           flx(ni+1)                          float32
!>           vel(ni+1)                          float32
!>           con(ni, nspcs)                     float32, Fortran order
!>   output: con(ni, nspcs)                     float32, updated in place
!>           vel(ni+1)                          float32, adjusted in place
program harness_vppm

   use CGRID_SPCS, only: set_species

   implicit none

   interface
      ! Copied from zadvyppm.F:183-201 (the non-isam, non-sens variant).
      ! CON is assumed-shape, so an explicit interface is mandatory.
      subroutine VPPM(NI, DT, DS, FLX, VEL, CON)
         integer, intent(in) :: NI
         real, intent(in) :: DT, DS(NI)
         real, intent(in) :: FLX(NI + 1)
         real, intent(inout) :: VEL(NI + 1)
         real, intent(inout) :: CON(:, :)
      end subroutine VPPM
   end interface

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios
   integer :: ni, nspcs
   real :: dt
   real, allocatable :: ds(:), flx(:), vel(:), con(:, :)

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_vppm <input.bin> <output.bin>'
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

   read (unit_in) ni, nspcs
   read (unit_in) dt

   allocate (ds(ni), flx(ni + 1), vel(ni + 1), con(ni, nspcs))

   read (unit_in) ds
   read (unit_in) flx
   read (unit_in) vel
   read (unit_in) con
   close (unit_in)

   ! Must happen before the first VPPM call: VPPM derives N_SPC_ADV from these
   ! counts once and SAVEs the result. set_species takes the number of
   ! transported species; VPPM adds one for rho*J, so pass nspcs - 1 to get
   ! N_SPC_ADV = nspcs, with slot nspcs holding rho*J.
   call set_species(nspcs - 1)

   call VPPM(ni, dt, ds, flx, vel, con)

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) con
   write (unit_out) vel
   close (unit_out)

end program harness_vppm
