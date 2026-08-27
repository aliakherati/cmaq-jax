!> Golden harness for HDIFF -- CMAQ's horizontal-diffusion driver.
!>
!>   usage: harness_hdiff <input.bin> <output.bin>
!>
!> Runs hdiff.F, hcdiff3d.F, deform.F and rho_j.F unmodified. Only the
!> meteorology source is replaced: interpolate_var returns the fields the
!> harness registered rather than reading an I/O API file.
!>
!> Two behaviours of the driver are what this golden exists to pin, because both
!> are easy to "clean up" in a rewrite and neither would announce itself:
!>
!>   * rho*J is NOT diffused. DIFF_MAP (hdiff.F:276-292) covers the transported
!>     species and has no +1 for density, unlike ADV_MAP. The last CGRID slot
!>     must come back unchanged.
!>   * the halo is frozen across sub-steps. HALO_* are filled once before the
!>     DO 344 loop from the initial mixing ratio while CONC is reloaded from
!>     CGRID every sub-step, so the zero-gradient boundary is exact only on the
!>     first sub-step. The number of sub-steps follows from the diffusivity, so
!>     a case with a large deformation exercises the drift.
!>
!> One call per process: hcdiff3d.F and deform.F both SAVE their grid geometry
!> on the first call.
!>
!>   input:  ncols, nrows, nlays, ntrns          (4 x int32)
!>           jdate, jtime                        (2 x int32)
!>           tstep(3)                            (3 x int32) HHMMSS
!>           dx1, dx2                            (2 x float64) cell size, m
!>           uhat_jd(ncols+1, nrows+1, nlays)    float32
!>           vhat_jd(ncols+1, nrows+1, nlays)    float32
!>           densa_j(ncols, nrows, nlays)        float32
!>           densa_j_bnd(nbndy, nlays)           float32
!>           msfd2(ncols+1, nrows+1)             float32
!>           cgrid(ncols, nrows, nlays, ntrns+1) float32, coupled; last slot rho*J
!>   output: cgrid(ncols, nrows, nlays, ntrns+1) float32, diffused in place
!>           nsteps                              int32, sub-steps taken
program harness_hdiff

   use HGRD_DEFN_STUB, only: set_hgrid
   use GRID_CONF, only: set_cell_size
   use VGRD_DEFN, only: set_vgrid
   use CGRID_SPCS, only: set_species
   use CENTRALIZED_IO_MODULE, only: cio_init, cio_put, cio_put_msfd2, cio_put_bndy
   use UTILIO_DEFN, only: set_file_vars

   implicit none

   interface
      subroutine HDIFF(CGRID, JDATE, JTIME, TSTEP)
         real, pointer :: CGRID(:, :, :, :)
         integer, intent(in) :: JDATE, JTIME
         integer, intent(in) :: TSTEP(3)
      end subroutine HDIFF
      subroutine HCDIFF3D(JDATE, JTIME, K11BAR, K22BAR, DT)
         integer, intent(in) :: JDATE, JTIME
         real, intent(out) :: K11BAR(:, :, :), K22BAR(:, :, :)
         real, intent(out) :: DT
      end subroutine HCDIFF3D
   end interface

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios
   integer :: ncols, nrows, nlays, ntrns, nspc, nbndy
   integer :: jdate, jtime, tstep(3), nsteps, l
   real, allocatable :: uhat(:, :, :), vhat(:, :, :), densa_j(:, :, :)
   real, allocatable :: densa_j_bnd(:, :), msfd2_in(:, :), faces(:)
   real, allocatable :: k11(:, :, :), k22(:, :, :)
   real, pointer :: cgrid(:, :, :, :) => null()
   real :: dt_stable, dtsec
   real(8) :: dx1, dx2

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_hdiff <input.bin> <output.bin>'
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

   read (unit_in) ncols, nrows, nlays, ntrns
   read (unit_in) jdate, jtime
   read (unit_in) tstep
   read (unit_in) dx1, dx2

   call set_hgrid(ncols, nrows)
   call set_cell_size(dx1, dx2)
   allocate (faces(nlays + 1))
   do l = 1, nlays + 1
      faces(l) = 1.0 - real(l - 1)/real(nlays)
   end do
   call set_vgrid(faces)
   call set_species(ntrns)
   nspc = ntrns + 1

   call cio_init(ncols, nrows, nlays)
   call set_file_vars(['UHAT_JD', 'VHAT_JD', 'DENSA_J'])

   nbndy = 2*(ncols + nrows + 2)
   allocate (uhat(ncols + 1, nrows + 1, nlays), vhat(ncols + 1, nrows + 1, nlays))
   allocate (densa_j(ncols, nrows, nlays), densa_j_bnd(nbndy, nlays))
   allocate (msfd2_in(ncols + 1, nrows + 1))
   read (unit_in) uhat
   read (unit_in) vhat
   read (unit_in) densa_j
   read (unit_in) densa_j_bnd
   read (unit_in) msfd2_in

   call cio_put('UHAT_JD', uhat)
   call cio_put('VHAT_JD', vhat)
   call cio_put('DENSA_J', densa_j)
   call cio_put_bndy('DENSA_J', densa_j_bnd)
   call cio_put_msfd2(msfd2_in)

   allocate (cgrid(ncols, nrows, nlays, nspc))
   read (unit_in) cgrid
   close (unit_in)

   ! The sub-step count is data-dependent and hdiff.F does not report it, so
   ! recompute it here exactly as hdiff.F:336-338 does. The port needs the same
   ! number to compare against, and it is a host-side decision there too.
   allocate (k11(ncols + 1, nrows + 1, nlays), k22(ncols + 1, nrows + 1, nlays))
   call HCDIFF3D(jdate, jtime, k11, k22, dt_stable)
   dtsec = real(3600*(tstep(2)/10000) + 60*(mod(tstep(2), 10000)/100) + mod(tstep(2), 100))
   nsteps = int(dtsec/dt_stable) + 1

   call HDIFF(cgrid, jdate, jtime, tstep)

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) cgrid
   write (unit_out) nsteps
   close (unit_out)

end program harness_hdiff
