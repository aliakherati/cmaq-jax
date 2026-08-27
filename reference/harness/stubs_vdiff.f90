!> Modules vdiffacmx.F needs beyond the shared stubs.
!>
!> The species maps, deposition-velocity carrier and emission array. All of
!> these are plain data upstream; the modules that *compute* them (depv/m3dry,
!> the DESID emission machinery) are outside this port's scope -- deposition
!> velocities and emission fluxes are inputs here, as meteorology is an input to
!> advection.
!>
!> The heterogeneous-HONO branches switch themselves off. `vdiffacmx.F:147`
!> declares HNO3_HIT/NO2_HIT/HONO_HIT locally, initialised to 0, and fills them
!> by searching the deposition species names (`vdiffacmx.F:262-270`). The generic
!> names here match none of them, so every index stays 0, no species index ever
!> equals it, and the driver takes its plain transport path throughout. That is
!> deliberate rather than incidental: those branches modify deposition
!> velocities using chemistry state, which is not vertical diffusion, and
!> excluding them makes the golden test exactly the scope being ported.

module VDIFF_MAP
   implicit none
   public

   integer :: N_SPC_DIFF = 0
   integer :: N_SPC_DEPV = 0
   integer, allocatable :: DIFF_MAP(:)   ! diffused species -> CGRID slot
   integer, allocatable :: DV2DF(:)      ! deposition species -> diffused index
   real, allocatable :: DD_CONV(:)       ! deposition unit conversion
   character(len=16), allocatable :: DV2DF_SPC(:)

contains

   !> One deposited species per diffused species, in the same order.
   subroutine set_vdiff_map(nspc)
      integer, intent(in) :: nspc
      integer :: i
      N_SPC_DIFF = nspc
      N_SPC_DEPV = nspc
      if (allocated(DIFF_MAP)) deallocate (DIFF_MAP, DV2DF, DD_CONV, DV2DF_SPC)
      allocate (DIFF_MAP(nspc), DV2DF(nspc), DD_CONV(nspc), DV2DF_SPC(nspc))
      DIFF_MAP = [(i, i=1, nspc)]
      DV2DF = [(i, i=1, nspc)]
      DD_CONV = 1.0
      DV2DF_SPC = 'SPC'
   end subroutine set_vdiff_map

end module VDIFF_MAP

module DEPV_DEFN
   implicit none
   public

   real, allocatable :: DEPV(:, :, :)   ! (nspc, ncols, nrows) [m/s]
   real, allocatable :: PLDV(:, :, :)   ! (nspc, ncols, nrows) emission flux
   logical :: ABFLUX = .false.
   logical :: MGN_ONLN_DEP = .false.

   !> Component-flux array length, from DEPVVARS.F:54. Only ABFLUX
   !> (bidirectional NH3) writes into it, and that is off here.
   integer, parameter :: LCMP = 8

contains

   subroutine depv_alloc(nspc, ncols, nrows)
      integer, intent(in) :: nspc, ncols, nrows
      if (allocated(DEPV)) deallocate (DEPV, PLDV)
      allocate (DEPV(nspc, ncols, nrows), PLDV(nspc, ncols, nrows))
      DEPV = 0.0
      PLDV = 0.0
   end subroutine depv_alloc

end module DEPV_DEFN

module DESID_VARS
   implicit none
   public
   integer :: DESID_LAYS = 1
   real, allocatable :: VDEMIS_DIFF(:, :, :, :)   ! (nspc, lays, ncols, nrows)
contains
   subroutine desid_alloc(nspc, nlays, ncols, nrows)
      integer, intent(in) :: nspc, nlays, ncols, nrows
      DESID_LAYS = nlays
      if (allocated(VDEMIS_DIFF)) deallocate (VDEMIS_DIFF)
      allocate (VDEMIS_DIFF(nspc, nlays, ncols, nrows))
      VDEMIS_DIFF = 0.0
   end subroutine desid_alloc
end module DESID_VARS

module DESID_PARAM_MODULE
   implicit none
   public
   integer :: DESID_N_SRM = 1
end module DESID_PARAM_MODULE

module VDIFF_DIAG
   implicit none
   public
   logical :: VDIFFDIAG = .false.
   real, allocatable :: NLPCR_MEAN(:, :)
end module VDIFF_DIAG

module BDSNP_MOD
   implicit none
   public
contains
   !> Never reached: MGN_ONLN_DEP is false.
   subroutine GET_N_DEP(spc, flux, c, r)
      character(len=*), intent(in) :: spc
      real, intent(in) :: flux
      integer, intent(in) :: c, r
      integer :: ignored
      ignored = len(spc) + c + r + int(flux)
   end subroutine GET_N_DEP
end module BDSNP_MOD
