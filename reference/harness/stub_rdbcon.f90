! Stand-in for RDBCON, CMAQ's boundary-concentration reader.
!
! The real rdbcon.F opens the BC file through the I/O API, maps its variables
! onto the advected species via advbc_map.F, applies ICBC scale factors, and
! couples the result with the Jacobian over the squared map-scale factor. All of
! that is I/O and bookkeeping -- none of it is advection -- and it drags in the
! aerosol modules (AERO_DATA, RUNTIME_VARS) for the wet/dry second-moment
! handling.
!
! So the boundary values arrive here already coupled, exactly as rdbcon.F would
! have left them, and this routine just hands back the requested layer. Every
! line of the actual advection chain -- hadvppm.F, x_ppm.F, y_ppm.F, hcontvel.F,
! zfdbc.f, hppm.F -- still runs unmodified.

module BCON_STORE
   implicit none
   public

   !> (NBNDY, N_SPC_ADV, NLAYS), in coupled transport units.
   real, allocatable :: bcon_data(:, :, :)

contains

   subroutine bcon_init(nbndy, nspc_adv, nlays)
      integer, intent(in) :: nbndy, nspc_adv, nlays
      if (allocated(bcon_data)) deallocate (bcon_data)
      allocate (bcon_data(nbndy, nspc_adv, nlays))
      bcon_data = 0.0
   end subroutine bcon_init

end module BCON_STORE

subroutine RDBCON(FDATE, FTIME, TSTEP, LVL, BCON, L_WRITE_WARNING)

   use BCON_STORE, only: bcon_data

   implicit none

   integer, intent(in) :: FDATE, FTIME, TSTEP, LVL
   real, intent(out) :: BCON(:, :)
   logical, intent(inout) :: L_WRITE_WARNING

   integer :: ignored

   ! The real RDBCON interpolates in time; the harness holds the boundary field
   ! fixed, so the date and step are not consulted.
   ignored = FDATE + FTIME + TSTEP

   if (.not. allocated(bcon_data)) then
      write (*, '(a)') 'RDBCON stub: bcon_init was never called'
      stop 1
   end if
   if (LVL < 1 .or. LVL > size(bcon_data, 3)) then
      write (*, '(a,i0)') 'RDBCON stub: layer out of range: ', LVL
      stop 1
   end if

   BCON = bcon_data(1:size(BCON, 1), 1:size(BCON, 2), LVL)
   L_WRITE_WARNING = .false.

end subroutine RDBCON
