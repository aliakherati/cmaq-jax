!> Golden harness for HCDIFF3D -- the horizontal eddy diffusivity, and the
!> wind deformation it is built from.
!>
!>   usage: harness_hcdiff3d <input.bin> <output.bin>
!>
!> Runs hcdiff3d.F and deform.F unmodified. Only the meteorology source is
!> replaced: interpolate_var returns the fields the harness registered rather
!> than reading an I/O API file.
!>
!> Emits DEFORM3D alongside K11BAR/K22BAR, so the two stages can be compared
!> separately. deform.F does not return it to a caller, so the harness recomputes
!> it by calling DEFORM directly -- the same call hcdiff3d.F makes internally,
!> with the same registered winds, so it is the same array rather than a
!> reimplementation.
!>
!> Both boundary conventions are visible in the output and they are different
!> edges, which is easy to conflate:
!>   * deform.F:337-343 zeroes DEFORM3D over the full (NCOLS+1, NROWS+1). Where
!>     deformation is zero the diffusivity is NOT zero -- KHD = max(KHMIN, 0)
!>     leaves the KHMIN floor.
!>   * hcdiff3d.F:216,226 then zeroes K11BAR(:,NROWS+1) and K22BAR(NCOLS+1,:).
!>
!> One call per process: hcdiff3d.F SAVEs DX1/DX2/KHA/ACOEF and its loop bounds
!> on the first call, so a second call with a different grid silently reuses the
!> first one's geometry.
!>
!>   input:  ncols, nrows, nlays                 (3 x int32)
!>           jdate, jtime                        (2 x int32)
!>           dx1, dx2                            (2 x float64) cell size, m
!>           uhat_jd(ncols+1, nrows+1, nlays)    float32   contravariant u * rhoJ
!>           vhat_jd(ncols+1, nrows+1, nlays)    float32   contravariant v * rhoJ
!>           densa_j(ncols, nrows, nlays)        float32   rho * J
!>           densa_j_bnd(nbndy, nlays)           float32   rho * J on the halo ring,
!>                                                         nbndy = 2*(ncols+nrows+2)
!>           msfd2(ncols+1, nrows+1)             float32   map scale factor^2, dot
!>   output: deform(ncols+1, nrows+1, nlays)     float32
!>           k11bar(ncols+1, nrows+1, nlays)     float32
!>           k22bar(ncols+1, nrows+1, nlays)     float32
!>           dt                                  float32   stability time step
program harness_hcdiff3d

   use HGRD_DEFN_STUB, only: set_hgrid
   use GRID_CONF, only: set_cell_size
   use VGRD_DEFN, only: set_vgrid
   use CENTRALIZED_IO_MODULE, only: cio_init, cio_put, cio_put_msfd2, cio_put_bndy
   use UTILIO_DEFN, only: set_file_vars

   implicit none

   interface
      subroutine HCDIFF3D(JDATE, JTIME, K11BAR, K22BAR, DT)
         integer, intent(in) :: JDATE, JTIME
         real, intent(out) :: K11BAR(:, :, :), K22BAR(:, :, :)
         real, intent(out) :: DT
      end subroutine HCDIFF3D
      subroutine DEFORM(JDATE, JTIME, DEFORM3D)
         integer, intent(in) :: JDATE, JTIME
         real, intent(out) :: DEFORM3D(:, :, :)
      end subroutine DEFORM
   end interface

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios
   integer :: ncols, nrows, nlays, jdate, jtime
   real, allocatable :: uhat(:, :, :), vhat(:, :, :), densa_j(:, :, :)
   real, allocatable :: msfd2_in(:, :), faces(:), densa_j_bnd(:, :)
   integer :: nbndy
   real, allocatable :: k11bar(:, :, :), k22bar(:, :, :), deform3d(:, :, :)
   real :: dt
   real(8) :: dx1, dx2
   integer :: l

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_hcdiff3d <input.bin> <output.bin>'
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

   read (unit_in) ncols, nrows, nlays
   read (unit_in) jdate, jtime
   read (unit_in) dx1, dx2

   call set_hgrid(ncols, nrows)
   call set_cell_size(dx1, dx2)
   ! hcdiff3d.F needs NLAYS but no actual vertical structure; a uniform column
   ! is enough, and nothing here reads the face values.
   allocate (faces(nlays + 1))
   do l = 1, nlays + 1
      faces(l) = 1.0 - real(l - 1)/real(nlays)
   end do
   call set_vgrid(faces)

   call cio_init(ncols, nrows, nlays)
   call set_file_vars(['UHAT_JD', 'VHAT_JD', 'DENSA_J'])

   allocate (uhat(ncols + 1, nrows + 1, nlays), vhat(ncols + 1, nrows + 1, nlays))
   allocate (densa_j(ncols, nrows, nlays), msfd2_in(ncols + 1, nrows + 1))
   nbndy = 2*(ncols + nrows + 2)
   allocate (densa_j_bnd(nbndy, nlays))
   read (unit_in) uhat
   read (unit_in) vhat
   read (unit_in) densa_j
   read (unit_in) densa_j_bnd
   read (unit_in) msfd2_in
   close (unit_in)

   call cio_put('UHAT_JD', uhat)
   call cio_put('VHAT_JD', vhat)
   call cio_put('DENSA_J', densa_j)
   ! deform.F takes the non-WINDOW path, which reads the interior and the halo
   ! ring separately and reassembles them (deform.F:250-292). The halo density
   ! is a real input here, not a convenience: it divides the contravariant wind
   ! at the domain-edge faces.
   call cio_put_bndy('DENSA_J', densa_j_bnd)
   call cio_put_msfd2(msfd2_in)

   allocate (k11bar(ncols + 1, nrows + 1, nlays))
   allocate (k22bar(ncols + 1, nrows + 1, nlays))
   allocate (deform3d(ncols + 1, nrows + 1, nlays))

   call DEFORM(jdate, jtime, deform3d)
   call HCDIFF3D(jdate, jtime, k11bar, k22bar, dt)

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) deform3d
   write (unit_out) k11bar
   write (unit_out) k22bar
   write (unit_out) dt
   close (unit_out)

end program harness_hcdiff3d
