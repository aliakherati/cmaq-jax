!> Golden harness for EDDYX -- the vertical eddy diffusivity.
!>
!>   usage: harness_eddyx <input.bin> <output.bin>
!>
!> Runs eddyx.F unmodified against a minimal ASX_DATA_MOD (harness/stubs_asx.f90)
!> holding just the twelve met arrays it reads.
!>
!> The parameterization has three regimes and the cases exist to separate them:
!> Monin-Obukhov similarity below the PBL top, a Richardson-number mixing length
!> above it, and a moist correction wherever cloud water is present. Above the
!> PBL the surface term is zero and only the free-atmosphere term survives.
!>
!> EDDYV is returned for layers 1..NLAYS-1 -- the diffusivity lives on layer
!> *interfaces*, so the top layer has none.
!>
!>   input:  ncols, nrows, nlays              (3 x int32)
!>           cstaguv                          (1 x int32) 1 = C-staggered winds
!>           pbl(ncols,nrows)                 float32
!>           ustar(ncols,nrows)               float32
!>           moli(ncols,nrows)                float32   1/L, sign gives stability
!>           zf(ncols,nrows,nlays)            float32
!>           zh(ncols,nrows,nlays)            float32
!>           kzmin(ncols,nrows,nlays)         float32
!>           thetav(ncols,nrows,nlays)        float32
!>           ta(ncols,nrows,nlays)            float32
!>           qv(ncols,nrows,nlays)            float32
!>           qc(ncols,nrows,nlays)            float32
!>           uwind(ncols+1,nrows+1,nlays)     float32
!>           vwind(ncols+1,nrows+1,nlays)     float32
!>   output: eddyv(ncols,nrows,nlays)         float32
program harness_eddyx

   use HGRD_DEFN_STUB, only: set_hgrid
   use VGRD_DEFN, only: set_vgrid
   use ASX_DATA_MOD, only: Met_Data, met_alloc, CSTAGUV

   implicit none

   interface
      subroutine EDDYX(EDDYV)
         real, intent(out) :: EDDYV(:, :, :)
      end subroutine EDDYX
   end interface

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios, ncols, nrows, nlays, cstag, k
   real, allocatable :: eddyv(:, :, :), faces(:)

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_eddyx <input.bin> <output.bin>'
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
   read (unit_in) cstag
   CSTAGUV = (cstag /= 0)

   call set_hgrid(ncols, nrows)
   allocate (faces(nlays + 1))
   do k = 1, nlays + 1
      faces(k) = 1.0 - real(k - 1)/real(nlays)
   end do
   call set_vgrid(faces)

   call met_alloc(ncols, nrows, nlays)
   read (unit_in) Met_Data%PBL
   read (unit_in) Met_Data%USTAR
   read (unit_in) Met_Data%MOLI
   read (unit_in) Met_Data%ZF
   read (unit_in) Met_Data%ZH
   read (unit_in) Met_Data%KZMIN
   read (unit_in) Met_Data%THETAV
   read (unit_in) Met_Data%TA
   read (unit_in) Met_Data%QV
   read (unit_in) Met_Data%QC
   read (unit_in) Met_Data%UWIND
   read (unit_in) Met_Data%VWIND
   close (unit_in)

   allocate (eddyv(ncols, nrows, nlays))
   eddyv = 0.0
   call EDDYX(eddyv)

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) eddyv
   close (unit_out)

end program harness_eddyx
