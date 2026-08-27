! Minimal stand-ins for the CMAQ modules the advection code references.
!
! The point of this file is that the vendored Fortran compiles UNMODIFIED.
! Nothing here does arithmetic on concentrations; it supplies names, dimensions
! and I/O API bookkeeping so the real science routines can run.
!
! Provide only what the code actually touches. If a symbol is missing the link
! will say so; do not add speculative ones.
!
! The stencil-exchange layer is NOT stubbed -- CMAQ ships a serial no-op
! implementation under STENEX/noop, vendored in reference/stenex/, and the
! Makefile points the SUBST_* macros at it exactly as a serial CMAQ build does.

!-----------------------------------------------------------------------
! UTILIO_DEFN: in CMAQ this re-exports the I/O API. The advection code uses
! an error exit, a message printer, the date/time helpers, and two file
! queries.
!-----------------------------------------------------------------------
module UTILIO_DEFN
   use iso_fortran_env, only: error_unit
   implicit none
   public

   integer, parameter :: XSTAT0 = 0
   integer, parameter :: XSTAT1 = 1
   integer, parameter :: XSTAT2 = 2
   integer, parameter :: XSTAT3 = 3

   ! Grid-type codes from PARMS3.EXT. LATGRD3 selects the degrees-to-metres
   ! branch at x_ppm.F:213.
   integer, parameter :: LATGRD3 = 1
   integer, parameter :: LAMGRD3 = 2

   ! I/O API file-description commons, filled by DESC3. hcontvel.F reads these
   ! to decide whether C-staggered winds exist.
   integer :: NVARS3D = 0
   character(len=16) :: VNAME3D(128) = ' '

   ! The variable list DESC3 reports. Held separately from VNAME3D because
   ! hcontvel.F:160 blanks VNAME3D immediately before calling DESC3, so a
   ! caller cannot simply pre-set the common -- DESC3 has to repopulate it on
   ! every call, exactly as the real I/O API does.
   integer :: STUB_FILE_NVARS = 0
   character(len=16) :: STUB_FILE_VNAMES(128) = ' '

   !> Unit the model writes its log to. The real one comes back from the I/O
   !> API's INIT3; pointing it at stderr keeps diagnostics away from any data
   !> a harness writes to stdout.
   integer :: LOGDEV = error_unit

   ! What LSTEPF reports as the last step on file. Far enough ahead that
   ! hcontvel.F's REVERT check never trips.
   integer :: STUB_LAST_DATE = 2099365
   integer :: STUB_LAST_TIME = 235959

contains

   subroutine M3EXIT(caller, jdate, jtime, msg, status)
      character(len=*), intent(in) :: caller
      integer, intent(in) :: jdate, jtime
      character(len=*), intent(in) :: msg
      integer, intent(in) :: status
      write (error_unit, '(a)') 'M3EXIT in '//trim(caller)//': '//trim(msg)
      write (error_unit, '(a,i0,a,i0,a,i0)') '  jdate=', jdate, ' jtime=', jtime, &
         ' status=', status
      stop 1
   end subroutine M3EXIT

   subroutine M3MESG(msg)
      character(len=*), intent(in) :: msg
      write (error_unit, '(a)') trim(msg)
   end subroutine M3MESG

   subroutine M3WARN(caller, jdate, jtime, msg)
      character(len=*), intent(in) :: caller
      integer, intent(in) :: jdate, jtime
      character(len=*), intent(in) :: msg
      write (error_unit, '(a,i0,1x,i0)') 'M3WARN in '//trim(caller)//': ' &
         //trim(msg)//' at ', jdate, jtime
   end subroutine M3WARN

   !> HHMMSS -> seconds. Sign is carried on the whole value, as in the I/O API.
   integer function TIME2SEC(hhmmss)
      integer, intent(in) :: hhmmss
      integer :: t
      t = abs(hhmmss)
      TIME2SEC = (t/10000)*3600 + mod(t/100, 100)*60 + mod(t, 100)
      if (hhmmss < 0) TIME2SEC = -TIME2SEC
   end function TIME2SEC

   !> Seconds -> HHMMSS. Hours are not wrapped at 24; the I/O API allows
   !> durations longer than a day.
   integer function SEC2TIME(seconds)
      integer, intent(in) :: seconds
      integer :: s
      s = abs(seconds)
      SEC2TIME = (s/3600)*10000 + mod(s/60, 60)*100 + mod(s, 60)
      if (seconds < 0) SEC2TIME = -SEC2TIME
   end function SEC2TIME

   !> Advance a Julian date (YYYYDDD) and time (HHMMSS) by a duration.
   subroutine NEXTIME(jdate, jtime, dtime)
      integer, intent(inout) :: jdate, jtime
      integer, intent(in) :: dtime
      integer :: total, days

      total = TIME2SEC(jtime) + TIME2SEC(dtime)
      days = total/86400
      total = total - days*86400
      if (total < 0) then
         total = total + 86400
         days = days - 1
      end if
      jtime = SEC2TIME(total)
      call advance_date(jdate, days)
   end subroutine NEXTIME

   subroutine advance_date(jdate, days)
      integer, intent(inout) :: jdate
      integer, intent(in) :: days
      integer :: year, doy, n, len_year

      if (days == 0) return
      year = jdate/1000
      doy = mod(jdate, 1000) + days
      do
         len_year = 365
         if (leap_year(year)) len_year = 366
         if (doy > len_year) then
            doy = doy - len_year
            year = year + 1
         else if (doy < 1) then
            year = year - 1
            n = 365
            if (leap_year(year)) n = 366
            doy = doy + n
         else
            exit
         end if
      end do
      jdate = year*1000 + doy
   end subroutine advance_date

   logical function leap_year(year)
      integer, intent(in) :: year
      leap_year = (mod(year, 4) == 0 .and. mod(year, 100) /= 0) .or. mod(year, 400) == 0
   end function leap_year

   !> Last date/time available on a file. Reported far in the future so the
   !> REVERT branch at hcontvel.F:208 never fires.
   subroutine LSTEPF(fname, ldate, ltime)
      character(len=*), intent(in) :: fname
      integer, intent(out) :: ldate, ltime
      character(len=1) :: ignored
      ignored = fname(1:1)
      ldate = STUB_LAST_DATE
      ltime = STUB_LAST_TIME
   end subroutine LSTEPF

   !> Declare which variables a file contains. Call before running anything
   !> that queries a file description.
   subroutine set_file_vars(names)
      character(len=*), intent(in) :: names(:)
      integer :: i
      STUB_FILE_NVARS = size(names)
      STUB_FILE_VNAMES = ' '
      do i = 1, STUB_FILE_NVARS
         STUB_FILE_VNAMES(i) = names(i)
      end do
   end subroutine set_file_vars

   !> File description query. Populates the NVARS3D/VNAME3D commons, which is
   !> how hcontvel.F:167 discovers whether UWINDC is present and therefore
   !> whether to take the C-staggered path.
   logical function DESC3(fname)
      character(len=*), intent(in) :: fname
      character(len=1) :: ignored
      ignored = fname(1:1)
      NVARS3D = STUB_FILE_NVARS
      VNAME3D = STUB_FILE_VNAMES
      DESC3 = .true.
   end function DESC3

   !> Position of `name` in `list`, or 0. The I/O API's INDEX1.
   integer function INDEX1(name, n, list)
      character(len=*), intent(in) :: name
      integer, intent(in) :: n
      character(len=*), intent(in) :: list(:)
      integer :: i
      INDEX1 = 0
      do i = 1, n
         if (trim(list(i)) == trim(name)) then
            INDEX1 = i
            return
         end if
      end do
   end function INDEX1

end module UTILIO_DEFN

!-----------------------------------------------------------------------
! PA_DEFN: process-analysis switches. All default .FALSE. so the budget and
! IPR blocks are skipped; the hppm harness turns the budget flags on
! explicitly because those outputs are part of its golden.
!-----------------------------------------------------------------------
module PA_DEFN
   implicit none
   public

   logical :: BUDGET_DIAG = .false.
   logical :: BUDGET_HPPM = .false.
   logical :: LIPR = .false.
   logical :: COUPLE_WRF = .true.
end module PA_DEFN

!-----------------------------------------------------------------------
! CGRID_SPCS: the species layout of the concentration array.
!
! CMAQ blocks CGRID as gas | aerosol | non-reactive | tracer, with each block
! carrying a map from its transported subset back to CGRID indices. The
! harness collapses everything into the gas block: only the sum and the maps
! matter to the advection code, never the split.
!-----------------------------------------------------------------------
module CGRID_SPCS
   implicit none
   public

   integer :: N_GC_TRNS = 0
   integer :: N_AE_TRNS = 0
   integer :: N_NR_TRNS = 0
   integer :: N_TR_ADV = 0

   integer :: N_GC_SPC = 0
   integer :: N_GC_SPCD = 0
   integer :: N_AE_SPC = 0
   integer :: N_NR_SPC = 0
   integer :: N_TR_SPC = 0
   integer :: NSPCSD = 0

   integer :: GC_STRT = 1
   integer :: AE_STRT = 1
   integer :: NR_STRT = 1
   integer :: TR_STRT = 1

   !> Index of air density (rho*J) within CGRID. CMAQ parks it in the slot just
   !> past the gas species, and advects it as an extra "species" so that
   !> transport conserves mass.
   integer :: RHOJ_LOC = 1

   !> Per-CGRID-slot flag: is this an aerosol species? x_ppm.F reads it only
   !> inside the budget block, to pick the unit conversion. The harness carries
   !> gases only, so it is all .FALSE.
   logical, allocatable :: CGRID_MASK_AERO(:)

   integer, allocatable :: GC_TRNS_MAP(:)
   integer, allocatable :: AE_TRNS_MAP(:)
   integer, allocatable :: NR_TRNS_MAP(:)
   integer, allocatable :: TR_ADV_MAP(:)

   !> Tracer diffusion map. hdiff.F builds DIFF_MAP from the same TRNS counts
   !> advection uses, but takes tracers through N_TR_DIFF rather than
   !> N_TR_ADV -- the two are separate namelist selections upstream.
   integer :: N_TR_DIFF = 0
   integer, allocatable :: TR_DIFF_MAP(:)

contains

   !> Lay out `ntrns` transported species followed by the rho*J slot, so that
   !> N_SPC_ADV = ntrns + 1 and CGRID has ntrns + 1 slots in the same order the
   !> advection code will build ADV_MAP.
   subroutine set_species(ntrns)
      integer, intent(in) :: ntrns
      integer :: i

      if (ntrns < 1) then
         write (*, '(a)') 'set_species: need at least one transported species'
         stop 1
      end if

      N_GC_TRNS = ntrns
      N_AE_TRNS = 0
      N_NR_TRNS = 0
      N_TR_ADV = 0
      N_TR_DIFF = 0

      N_GC_SPC = ntrns
      N_GC_SPCD = ntrns + 1     ! gas block plus the rho*J slot
      N_AE_SPC = 0
      N_NR_SPC = 0
      N_TR_SPC = 0
      NSPCSD = N_GC_SPCD

      GC_STRT = 1
      AE_STRT = N_GC_SPCD + 1
      NR_STRT = AE_STRT
      TR_STRT = AE_STRT
      RHOJ_LOC = N_GC_SPCD

      if (allocated(CGRID_MASK_AERO)) deallocate (CGRID_MASK_AERO)
      allocate (CGRID_MASK_AERO(NSPCSD))
      CGRID_MASK_AERO = .false.

      if (allocated(GC_TRNS_MAP)) deallocate (GC_TRNS_MAP)
      if (allocated(AE_TRNS_MAP)) deallocate (AE_TRNS_MAP)
      if (allocated(NR_TRNS_MAP)) deallocate (NR_TRNS_MAP)
      if (allocated(TR_ADV_MAP)) deallocate (TR_ADV_MAP)
      if (allocated(TR_DIFF_MAP)) deallocate (TR_DIFF_MAP)
      allocate (GC_TRNS_MAP(ntrns), AE_TRNS_MAP(1), NR_TRNS_MAP(1), TR_ADV_MAP(1))
      allocate (TR_DIFF_MAP(1))
      GC_TRNS_MAP = [(i, i=1, ntrns)]
      AE_TRNS_MAP = 0
      NR_TRNS_MAP = 0
      TR_ADV_MAP = 0
      TR_DIFF_MAP = 0
   end subroutine set_species

end module CGRID_SPCS
