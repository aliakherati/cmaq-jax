!> Golden harness for VDIFFACMX -- the ACM2 vertical-diffusion driver.
!>
!>   usage: harness_vdiff <input.bin> <output.bin>
!>
!> Runs vdiffacmx.F, tri.F and matrix1.F unmodified. Deposition velocities and
!> emission fluxes are supplied as inputs rather than computed, which is this
!> port's scope boundary (see docs/plans/PLAN-vdiff.md).
!>
!> The heterogeneous-HONO species indices in the VDIFF_MAP stub are -1, so no
!> species matches them and the driver takes its plain transport path
!> throughout. That makes the golden test exactly the code being ported rather
!> than a superset of it.
!>
!> Note vdiffacmx.F MODIFIES SEDDY in place: the convective stage removes a
!> fraction FNL of the eddy diffusivity inside the CBL and carries it
!> non-locally instead (vdiffacmx.F:493-499). The scaled array is returned so a
!> port can be checked against that split, which is the essence of ACM2 and easy
!> to miss.
!>
!>   input:  ncols, nrows, nlays, nspc         (4 x int32)
!>           dtsec                             float32   sync step
!>           convct(ncols,nrows)               int32     1 = convective column
!>           lpbl(ncols,nrows)                 int32     layer index of PBL top
!>           pbl, hol, dens1, rdepvht          float32   (ncols,nrows) each
!>           zf(ncols,nrows,nlays)             float32   layer face heights
!>           zh(ncols,nrows,nlays)             float32   layer middle heights
!>           seddy(nlays,ncols,nrows)          float32   Kz, layer-first
!>           depv(nspc,ncols,nrows)            float32   deposition velocity
!>           pldv(nspc,ncols,nrows)            float32   emission flux at surface
!>           vdemis(nspc,nlays,ncols,nrows)    float32   layered emissions
!>           cngrd(nspc,nlays,ncols,nrows)     float32   concentrations
!>   output: cngrd(nspc,nlays,ncols,nrows)     float32   diffused
!>           ddep(nspc,ncols,nrows)            float32   accumulated dry dep
!>           seddy(nlays,ncols,nrows)          float32   after the ACM2 split
program harness_vdiff

   use HGRD_DEFN_STUB, only: set_hgrid
   use VGRD_DEFN, only: set_vgrid
   use CGRID_SPCS, only: set_species
   use ASX_DATA_MOD, only: Met_Data, met_alloc
   use VDIFF_MAP, only: set_vdiff_map
   use DEPV_DEFN, only: DEPV, PLDV, depv_alloc
   use DESID_VARS, only: VDEMIS_DIFF, desid_alloc

   implicit none

   interface
      subroutine VDIFFACMX(DTSEC, SEDDY, DDEP, ICMP, CNGRD)
         real, intent(in) :: DTSEC
         real, intent(inout) :: SEDDY(:, :, :)
         real, intent(inout) :: DDEP(:, :, :)
         real, intent(inout) :: ICMP(:, :, :)
         real, intent(inout) :: CNGRD(:, :, :, :)
      end subroutine VDIFFACMX
   end interface

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios, ncols, nrows, nlays, nspc, k
   integer, allocatable :: iflag(:, :)
   real, allocatable :: seddy(:, :, :), ddep(:, :, :), icmp(:, :, :)
   real, allocatable :: cngrd(:, :, :, :), faces(:)
   real :: dtsec

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_vdiff <input.bin> <output.bin>'
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

   read (unit_in) ncols, nrows, nlays, nspc
   read (unit_in) dtsec

   call set_hgrid(ncols, nrows)
   allocate (faces(nlays + 1))
   do k = 1, nlays + 1
      faces(k) = 1.0 - real(k - 1)/real(nlays)
   end do
   call set_vgrid(faces)
   call set_species(nspc)
   call set_vdiff_map(nspc)
   call met_alloc(ncols, nrows, nlays)
   call depv_alloc(nspc, ncols, nrows)
   call desid_alloc(nspc, nlays, ncols, nrows)

   allocate (iflag(ncols, nrows))
   read (unit_in) iflag
   Met_Data%CONVCT = (iflag /= 0)
   read (unit_in) Met_Data%LPBL
   read (unit_in) Met_Data%PBL
   read (unit_in) Met_Data%HOL
   read (unit_in) Met_Data%DENS1
   read (unit_in) Met_Data%RDEPVHT
   read (unit_in) Met_Data%ZF
   read (unit_in) Met_Data%ZH

   allocate (seddy(nlays, ncols, nrows))
   read (unit_in) seddy
   read (unit_in) DEPV
   read (unit_in) PLDV
   read (unit_in) VDEMIS_DIFF

   allocate (cngrd(nspc, nlays, ncols, nrows))
   read (unit_in) cngrd
   close (unit_in)

   allocate (ddep(nspc, ncols, nrows), icmp(nspc, ncols, nrows))
   ddep = 0.0
   icmp = 0.0

   call VDIFFACMX(dtsec, seddy, ddep, icmp, cngrd)

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) cngrd
   write (unit_out) ddep
   write (unit_out) seddy
   close (unit_out)

end program harness_vdiff
