!> Golden harness for TRI -- the Thomas tridiagonal solver ACM2's local stage uses.
!>
!>   usage: harness_tri <input.bin> <output.bin>
!>
!> Runs tri.F unmodified. Pure linear algebra: no meteorology, no I/O API.
!>
!> One matrix, many right-hand sides. tri.F takes L/D/U once and solves every
!> species against them, which is how ACM2 uses it -- the factorisation is shared
!> across the whole column, not repeated per species.
!>
!> The matrix is stored as CMAQ stores it (tri.F:40-46):
!>
!>   [ D(1) U(1)  0    ...        ]
!>   [ L(2) D(2) U(2)  ...        ]  X = B
!>   [  0   L(3) D(3)  ...        ]
!>
!> so L(1) and U(NLAYS) are never referenced. The harness still reads them, and
!> the golden generator fills them with a poison value, so that a port which
!> accidentally uses them disagrees loudly rather than subtly.
!>
!>   input:  nlays, nspcs                (2 x int32)
!>           l(nlays)                    float32   subdiagonal, l(1) unused
!>           d(nlays)                    float32   diagonal
!>           u(nlays)                    float32   superdiagonal, u(nlays) unused
!>           b(nspcs, nlays)             float32   right-hand sides
!>   output: x(nspcs, nlays)             float32   solution
program harness_tri

   use VGRD_DEFN, only: set_vgrid
   use CGRID_SPCS, only: set_species

   implicit none

   interface
      subroutine TRI(L, D, U, B, X)
         real, intent(in) :: L(:), D(:), U(:)
         real, intent(in) :: B(:, :)
         real, intent(out) :: X(:, :)
      end subroutine TRI
   end interface

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios, nlays, nspcs, k
   real, allocatable :: l(:), d(:), u(:), b(:, :), x(:, :), faces(:)

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_tri <input.bin> <output.bin>'
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

   read (unit_in) nlays, nspcs

   ! tri.F takes NLAYS from VGRD_DEFN and its species count from CGRID_SPCS;
   ! neither needs real vertical structure here.
   allocate (faces(nlays + 1))
   do k = 1, nlays + 1
      faces(k) = 1.0 - real(k - 1)/real(nlays)
   end do
   call set_vgrid(faces)
   call set_species(nspcs)

   allocate (l(nlays), d(nlays), u(nlays), b(nspcs, nlays), x(nspcs, nlays))
   read (unit_in) l
   read (unit_in) d
   read (unit_in) u
   read (unit_in) b
   close (unit_in)

   x = 0.0
   call TRI(l, d, u, b, x)

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) x
   close (unit_out)

end program harness_tri
