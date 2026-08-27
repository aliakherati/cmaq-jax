!> Minimal ASX_DATA_MOD, for the EDDYX harness.
!>
!> The real module is 1154 lines of meteorology bookkeeping, land-surface
!> coupling and I/O. `eddyx.F` uses twelve of its arrays, three of its
!> parameters and one logical, so that is all this provides -- the same
!> principle as every other stub here: supply the names the vendored file needs
!> and nothing else, so that what is being tested is unambiguous.
!>
!> The three parameters are declared in *lower case* upstream
!> (`ASX_DATA_MOD.F:216-219`), which is easy to miss when searching a codebase
!> that is otherwise uppercase. Values are reproduced exactly.
module ASX_DATA_MOD
   implicit none
   public

   include "CONST.EXT"

   ! ASX_DATA_MOD.F:216-219. betah/gamah are the Dyer stability-function
   ! coefficients as WRF 3.6 PX uses them; karman is von Karman's constant.
   real, parameter :: betah = 5.0
   real, parameter :: gamah = 16.0
   real, parameter :: karman = 0.40

   !> Whether the winds are C-staggered. Chooses between two wind-shear
   !> stencils in eddyx.F:138-152, so a harness has to be able to set it.
   logical :: CSTAGUV = .true.

   type MET_Type
      ! 2-D surface fields.
      real, allocatable :: PBL(:, :)      ! PBL height [m]
      real, allocatable :: USTAR(:, :)    ! friction velocity [m/s]
      real, allocatable :: MOLI(:, :)     ! inverse Monin-Obukhov length [1/m]
      ! 3-D fields at layer faces / middles.
      real, allocatable :: ZF(:, :, :)     ! layer face height [m]
      real, allocatable :: ZH(:, :, :)     ! layer middle height [m]
      real, allocatable :: KZMIN(:, :, :)  ! minimum Kz [m2/s]
      real, allocatable :: THETAV(:, :, :) ! virtual potential temperature [K]
      real, allocatable :: TA(:, :, :)     ! air temperature [K]
      real, allocatable :: QV(:, :, :)     ! water vapour mixing ratio [kg/kg]
      real, allocatable :: QC(:, :, :)     ! cloud water mixing ratio [kg/kg]
      ! Dot-dimensioned winds, (NCOLS+1, NROWS+1, NLAYS) upstream.
      real, allocatable :: UWIND(:, :, :)
      real, allocatable :: VWIND(:, :, :)
      ! Additional fields vdiffacmx.F reads.
      real, allocatable :: DENS1(:, :)      ! layer-1 air density [kg/m3]
      real, allocatable :: RDEPVHT(:, :)    ! 1 / deposition height [1/m]
      real, allocatable :: HOL(:, :)        ! PBL height / Monin-Obukhov length
      integer, allocatable :: LPBL(:, :)    ! layer index of the PBL top
      logical, allocatable :: CONVCT(:, :)  ! is this column convective?
   end type MET_Type

   type(MET_Type) :: Met_Data

contains

   subroutine met_alloc(ncols, nrows, nlays)
      integer, intent(in) :: ncols, nrows, nlays
      allocate (Met_Data%PBL(ncols, nrows))
      allocate (Met_Data%USTAR(ncols, nrows))
      allocate (Met_Data%MOLI(ncols, nrows))
      allocate (Met_Data%ZF(ncols, nrows, nlays))
      allocate (Met_Data%ZH(ncols, nrows, nlays))
      allocate (Met_Data%KZMIN(ncols, nrows, nlays))
      allocate (Met_Data%THETAV(ncols, nrows, nlays))
      allocate (Met_Data%TA(ncols, nrows, nlays))
      allocate (Met_Data%QV(ncols, nrows, nlays))
      allocate (Met_Data%QC(ncols, nrows, nlays))
      allocate (Met_Data%UWIND(ncols + 1, nrows + 1, nlays))
      allocate (Met_Data%VWIND(ncols + 1, nrows + 1, nlays))
      allocate (Met_Data%DENS1(ncols, nrows))
      allocate (Met_Data%RDEPVHT(ncols, nrows))
      allocate (Met_Data%HOL(ncols, nrows))
      allocate (Met_Data%LPBL(ncols, nrows))
      allocate (Met_Data%CONVCT(ncols, nrows))
      Met_Data%DENS1 = 1.2
      Met_Data%RDEPVHT = 0.05
      Met_Data%HOL = -1.0
      Met_Data%LPBL = 1
      Met_Data%CONVCT = .false.
   end subroutine met_alloc

end module ASX_DATA_MOD
