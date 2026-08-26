! Minimal stand-ins for the CMAQ modules that hppm.F and vppm.F reference.
!
! The point of this file is that the two kernels compile UNMODIFIED. Their
! dependency surface is tiny -- an error exit, a couple of flags, and the
! serial form of one stencil-exchange call -- so nothing here needs the I/O
! API, netCDF, or MPI.
!
! Provide only what the kernels actually touch. If a symbol is missing the
! link will say so; do not add speculative ones.

!-----------------------------------------------------------------------
! UTILIO_DEFN: hppm.F and vppm.F use only M3EXIT and the XSTAT1 code.
!-----------------------------------------------------------------------
module UTILIO_DEFN
   use iso_fortran_env, only: error_unit
   implicit none
   public

   integer, parameter :: XSTAT0 = 0
   integer, parameter :: XSTAT1 = 1
   integer, parameter :: XSTAT2 = 2
   integer, parameter :: XSTAT3 = 3

contains

   !> Abort. The real M3EXIT writes to the CMAQ log and stops; a non-zero
   !> exit status is what the Python harness needs to see.
   subroutine M3EXIT(caller, jdate, jtime, msg, status)
      character(len=*), intent(in) :: caller
      integer, intent(in) :: jdate, jtime
      character(len=*), intent(in) :: msg
      integer, intent(in) :: status
      write (error_unit, '(a)') 'M3EXIT in '//trim(caller)//': '//trim(msg)
      write (error_unit, '(a,i0,a,i0)') '  jdate=', jdate, ' jtime=', jtime
      stop 1
   end subroutine M3EXIT

end module UTILIO_DEFN

!-----------------------------------------------------------------------
! HGRD_DEFN: hppm.F has `USE HGRD_DEFN` but references nothing from it --
! the USE is vestigial. An empty module satisfies it.
!-----------------------------------------------------------------------
module HGRD_DEFN
   implicit none
   public
end module HGRD_DEFN

!-----------------------------------------------------------------------
! PA_DEFN: hppm.F reads BUDGET_HPPM to decide whether to fill the boundary
! flux outputs (hppm.F:451). We set it .TRUE. so the goldens capture them --
! with it .FALSE. those INTENT(OUT) arrays are left undefined.
!-----------------------------------------------------------------------
module PA_DEFN
   implicit none
   public

   logical :: BUDGET_DIAG = .true.
   logical :: BUDGET_HPPM = .true.
end module PA_DEFN

!-----------------------------------------------------------------------
! NOOP_MODULES: the serial stencil-exchange shim. hppm.F calls the cpp macro
! SUBST_HI_LO_BND_PE, which the Makefile defines to stub_hi_lo_bnd_pe.
!-----------------------------------------------------------------------
module NOOP_MODULES
   implicit none
   public

contains

   !> Serial case: this process owns the whole domain, so it is both the low
   !> and the high boundary in either orientation. Mirrors noop_hi_lo_bnd_pe
   !> in CCTM/src/STENEX/noop/noop_util_module.f.
   subroutine stub_hi_lo_bnd_pe(ori, lo, hi)
      character, intent(in) :: ori
      logical, intent(out) :: lo, hi
      character :: ignored
      ignored = ori
      lo = .true.
      hi = .true.
   end subroutine stub_hi_lo_bnd_pe

end module NOOP_MODULES

!-----------------------------------------------------------------------
! CGRID_SPCS: vppm.F computes
!     N_SPC_ADV = N_GC_TRNS + N_AE_TRNS + N_NR_TRNS + N_TR_ADV + 1
! once, on its first call, and SAVEs it. These are plain variables (not
! parameters) so the harness can set them from its input before that first
! call. The trailing +1 is the rho*J slot.
!-----------------------------------------------------------------------
module CGRID_SPCS
   implicit none
   public

   integer :: N_GC_TRNS = 0
   integer :: N_AE_TRNS = 0
   integer :: N_NR_TRNS = 0
   integer :: N_TR_ADV = 0

contains

   !> Set the species counts so that N_SPC_ADV comes out as `nspcs`.
   !> Everything lands in the gas-transport count; vppm.F only ever uses the
   !> sum, so the split does not matter.
   subroutine set_n_spc_adv(nspcs)
      integer, intent(in) :: nspcs
      if (nspcs < 1) then
         write (*, '(a)') 'set_n_spc_adv: nspcs must be >= 1'
         stop 1
      end if
      N_GC_TRNS = nspcs - 1
      N_AE_TRNS = 0
      N_NR_TRNS = 0
      N_TR_ADV = 0
   end subroutine set_n_spc_adv

end module CGRID_SPCS
