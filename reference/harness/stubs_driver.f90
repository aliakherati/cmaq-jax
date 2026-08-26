! Stand-ins for the CMAQ modules that the horizontal-advection *driver* needs.
!
! stubs.f90 covers what the bare PPM kernels reference. The driver chain --
! hadvppm.F -> x_ppm.F / y_ppm.F -> hcontvel.F -> hppm.F -- reaches further: it
! wants grid dimensions, the advected-species map, and meteorology.
!
! The meteorology is the point of this file. `interpolate_var` is CMAQ's single
! entry point for reading a met field at a given date/time; in the real model it
! pulls from I/O API files through a large caching layer. Here it is a lookup
! into a table the harness fills before calling anything, so every line of the
! actual advection code runs unmodified while the data comes from the test.
!
! Nothing here is science. If a routine in this file starts doing arithmetic on
! concentrations, it belongs in the vendored Fortran instead.

!-----------------------------------------------------------------------
! Horizontal grid. Dimensions are variables, not parameters, so one build
! serves every domain size the harness wants to exercise.
!-----------------------------------------------------------------------
module HGRD_DEFN_STUB
   implicit none
   public

   integer :: NCOLS = 0
   integer :: NROWS = 0
   integer :: NTHIK = 1
   integer :: NBNDY = 0
   integer :: GL_NCOLS = 0
   integer :: GL_NROWS = 0
   integer :: MY_NCOLS = 0
   integer :: MY_NROWS = 0

   ! Projection. GDTYP_GD = 2 is Lambert conformal, the usual CMAQ choice, for
   ! which x_ppm.F takes XCELL_GD directly as the cell width in metres. Setting
   ! it to LATGRD3 (= 1) instead selects the degrees-to-metres branch at
   ! x_ppm.F:213.
   integer :: GDTYP_GD = 2
   real(8) :: XCELL_GD = 12000.0d0
   real(8) :: YCELL_GD = 12000.0d0
   real(8) :: XORIG_GD = 0.0d0
   real(8) :: YORIG_GD = 0.0d0

contains

   subroutine set_hgrid(ncols_in, nrows_in)
      integer, intent(in) :: ncols_in, nrows_in
      NCOLS = ncols_in
      NROWS = nrows_in
      MY_NCOLS = ncols_in
      MY_NROWS = nrows_in
      GL_NCOLS = ncols_in
      GL_NROWS = nrows_in
      ! CMAQ's boundary array is a ring NTHIK cells deep around the domain.
      NBNDY = 2 * NTHIK * (NCOLS + NROWS + 2 * NTHIK)
   end subroutine set_hgrid

end module HGRD_DEFN_STUB

!-----------------------------------------------------------------------
! Vertical grid.
!-----------------------------------------------------------------------
module VGRD_DEFN
   implicit none
   public
   integer :: NLAYS = 1
end module VGRD_DEFN

!-----------------------------------------------------------------------
! GRID_CONF is CMAQ's umbrella over the two grid modules; most science
! routines USE this rather than either one directly.
!-----------------------------------------------------------------------
module GRID_CONF
   use HGRD_DEFN_STUB
   use VGRD_DEFN
   implicit none
   public
end module GRID_CONF

!-----------------------------------------------------------------------
! The name x_ppm.F and hcontvel.F actually USE.
!-----------------------------------------------------------------------
module HGRD_DEFN
   use HGRD_DEFN_STUB
   implicit none
   public
end module HGRD_DEFN

!-----------------------------------------------------------------------
! Meteorology source.
!
! `window` is CMAQ's flag for running on a sub-window of the met files; false
! means the domain matches the files, which is the case the harness sets up.
!-----------------------------------------------------------------------
module CENTRALIZED_IO_MODULE
   implicit none
   public

   integer, parameter :: MAX_FIELDS = 32
   integer, parameter :: NAME_LEN = 16

   logical :: window = .false.

   ! Boundary-condition file variable list, used by advbc_map.F. Unused by the
   ! driver harness, which passes BCON in directly.
   integer :: n_cio_bc_file_vars = 0
   character(len=NAME_LEN) :: cio_bc_file_var_name(1) = ' '

   ! Map-scale factor squared, read by x_ppm.F only inside the budget block.
   real, allocatable :: MSFX2(:, :)

   ! The field table. Every entry is stored at full 3-D extent; 2-D requests
   ! take a layer out of it.
   integer :: n_fields = 0
   character(len=NAME_LEN) :: field_name(MAX_FIELDS) = ' '
   real, allocatable :: field_data(:, :, :, :)   ! (col, row, lay, field)

   interface interpolate_var
      module procedure r_interpolate_var_2d
      module procedure r_interpolate_var_2db
      module procedure r_interpolate_var_3d
   end interface interpolate_var

contains

   subroutine cio_init(ncols, nrows, nlays)
      integer, intent(in) :: ncols, nrows, nlays
      if (allocated(field_data)) deallocate (field_data)
      ! One extra column and row: the staggered wind components live on cell
      ! faces, so UWINDC is dimensioned (NCOLS+1, NROWS).
      allocate (field_data(ncols + 1, nrows + 1, nlays, MAX_FIELDS))
      field_data = 0.0
      n_fields = 0
      field_name = ' '
      if (allocated(MSFX2)) deallocate (MSFX2)
      allocate (MSFX2(ncols, nrows))
      MSFX2 = 1.0
   end subroutine cio_init

   !> Register a field. Later registrations of the same name overwrite it.
   subroutine cio_put(vname, values)
      character(len=*), intent(in) :: vname
      real, intent(in) :: values(:, :, :)
      integer :: slot, nc, nr, nl

      slot = cio_slot(vname)
      if (slot == 0) then
         n_fields = n_fields + 1
         if (n_fields > MAX_FIELDS) then
            write (*, '(a)') 'cio_put: too many fields'
            stop 1
         end if
         slot = n_fields
         field_name(slot) = vname
      end if

      nc = size(values, 1)
      nr = size(values, 2)
      nl = size(values, 3)
      field_data(1:nc, 1:nr, 1:nl, slot) = values
   end subroutine cio_put

   integer function cio_slot(vname)
      character(len=*), intent(in) :: vname
      integer :: i
      cio_slot = 0
      do i = 1, n_fields
         if (trim(field_name(i)) == trim(vname)) then
            cio_slot = i
            return
         end if
      end do
   end function cio_slot

   !> Abort loudly on an unregistered name. A silent zero here would look like
   !> a calm day rather than a missing input, and the advection would quietly
   !> do nothing.
   integer function cio_require(vname)
      character(len=*), intent(in) :: vname
      cio_require = cio_slot(vname)
      if (cio_require == 0) then
         write (*, '(a)') 'interpolate_var: no such field registered: '//trim(vname)
         stop 1
      end if
   end function cio_require

   subroutine r_interpolate_var_2d(vname, date, time, data, scol, ecol, srow, erow, slay)
      character(len=*), intent(in) :: vname
      integer, intent(in) :: date, time
      real, intent(out) :: data(:, :)
      integer, intent(in), optional :: scol, ecol, srow, erow, slay
      integer :: slot, lay, ignored

      ignored = date + time
      if (present(scol) .or. present(ecol) .or. present(srow) .or. present(erow)) then
         write (*, '(a)') 'interpolate_var: sub-window requests are not supported by the stub'
         stop 1
      end if

      slot = cio_require(vname)
      lay = 1
      if (present(slay)) lay = slay
      data = field_data(1:size(data, 1), 1:size(data, 2), lay, slot)
   end subroutine r_interpolate_var_2d

   subroutine r_interpolate_var_2db(vname, date, time, data, type, lvl)
      character(len=*), intent(in) :: vname
      character(len=1), intent(in) :: type
      integer, intent(in) :: date, time
      real, intent(out) :: data(:, :)
      integer, intent(in), optional :: lvl
      integer :: ignored
      character :: ignored_type

      ignored = date + time
      ignored_type = type
      if (present(lvl)) ignored = ignored + lvl
      ! Boundary reads only happen on the CSTAGUV = .FALSE. path, which the
      ! driver harness does not exercise. Fail loudly rather than return zeros.
      write (*, '(a)') 'interpolate_var: boundary reads are not supported by the stub: '//trim(vname)
      stop 1
   end subroutine r_interpolate_var_2db

   subroutine r_interpolate_var_3d(vname, date, time, data, fname)
      character(len=*), intent(in) :: vname
      integer, intent(in) :: date, time
      real, intent(out) :: data(:, :, :)
      character(len=*), intent(in), optional :: fname
      integer :: slot, ignored

      ignored = date + time
      if (present(fname)) ignored = ignored + len(fname)
      slot = cio_require(vname)
      data = field_data(1:size(data, 1), 1:size(data, 2), 1:size(data, 3), slot)
   end subroutine r_interpolate_var_3d

end module CENTRALIZED_IO_MODULE

!-----------------------------------------------------------------------
! Runtime option flags, referenced by rdbcon.F. Present so the module
! resolves; the driver harness passes BCON in directly.
!-----------------------------------------------------------------------
module RUNTIME_VARS
   implicit none
   public
   logical :: BC_AERO_M2WET = .false.
   logical :: BC_AERO_M2USE = .false.
end module RUNTIME_VARS
