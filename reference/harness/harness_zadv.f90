!> Golden harness for ZADV -- CMAQ's vertical-advection driver.
!>
!>   usage: harness_zadv <input.bin> <output.bin>
!>
!> Runs zadvppmwrf.F and vppm.F unmodified. Only the meteorology source is
!> replaced: interpolate_var reads the DENSA_J the harness registered rather
!> than an I/O API file.
!>
!> ZADV reads DENSA_J twice -- once at the start of the sync step and once at
!> the end. Both are accepted here, but note that with FBLN fixed at 1.0
!> (zadvppmwrf.F:249) the end-of-step field feeds only a dead accumulator and
!> has no effect on the answer.
!>
!> One call per process: vppm.F sizes its work arrays and its non-uniform mesh
!> coefficients on the first call and SAVEs both.
!>
!>   input:  ncols, nrows, nlays, ntrns          (4 x int32)
!>           jdate, jtime                        (2 x int32)
!>           tstep(3)                            (3 x int32) HHMMSS
!>           x3face(nlays+1)                     float32   sigma faces
!>           rhojm1(ncols, nrows, nlays)         float32   met density, start
!>           rhojm2(ncols, nrows, nlays)         float32   met density, end
!>           cgrid(ncols, nrows, nlays, ntrns+1) float32
!>   output: cgrid(ncols, nrows, nlays, ntrns+1) float32, advected in place
program harness_zadv

   use HGRD_DEFN_STUB, only: set_hgrid
   use VGRD_DEFN, only: NLAYS, X3FACE_GD, set_vgrid
   use CGRID_SPCS, only: set_species
   use CENTRALIZED_IO_MODULE, only: cio_init, cio_put
   use UTILIO_DEFN, only: set_file_vars

   implicit none

   interface
      subroutine ZADV(CGRID, JDATE, JTIME, TSTEP)
         real, pointer :: CGRID(:, :, :, :)
         integer, intent(in) :: JDATE, JTIME
         integer, intent(in) :: TSTEP(3)
      end subroutine ZADV
   end interface

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios
   integer :: ncols, nrows, nlays_in, ntrns, nspc_adv
   integer :: jdate, jtime, tstep(3)
   real, allocatable :: faces(:), rhojm1(:, :, :), rhojm2(:, :, :)
   real, pointer :: cgrid(:, :, :, :) => null()

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_zadv <input.bin> <output.bin>'
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
   read (unit_in) jdate, jtime
   read (unit_in) tstep

   call set_hgrid(ncols, nrows)
   allocate (faces(nlays_in + 1))
   read (unit_in) faces
   call set_vgrid(faces)
   call set_species(ntrns)
   nspc_adv = ntrns + 1

   call cio_init(ncols, nrows, nlays_in)
   call set_file_vars(['DENSA_J'])

   allocate (rhojm1(ncols, nrows, nlays_in), rhojm2(ncols, nrows, nlays_in))
   read (unit_in) rhojm1
   read (unit_in) rhojm2
   ! The stub ignores the requested time, so both reads return the same field.
   ! Harmless: FBLN = 1.0 makes the end-of-step value unused anyway.
   call cio_put('DENSA_J', rhojm1)

   allocate (cgrid(ncols, nrows, nlays_in, nspc_adv))
   read (unit_in) cgrid
   close (unit_in)

   call ZADV(cgrid, jdate, jtime, tstep)

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) cgrid
   close (unit_out)

end program harness_zadv
