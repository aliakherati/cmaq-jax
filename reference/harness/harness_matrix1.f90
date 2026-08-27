!> Golden harness for MATRIX1 -- the ACM1 solver for the convective stage.
!>
!>   usage: harness_matrix1 <input.bin> <output.bin>
!>
!> Runs matrix1.F unmodified. The matrix is tridiagonal-plus-first-column, which
!> is what the ACM2 non-local plume produces: mass leaves the surface layer and
!> arrives directly in every layer of the convective boundary layer, so every row
!> couples to column 1.
!>
!>   [ B(1)  E(2)                     ]
!>   [ A(2)  B(2)  E(3)               ]  X = D
!>   [ A(3)        B(3)  E(4)         ]
!>   [ A(4)              B(4)   ...   ]
!>
!> Note the shape: A is a *column*, not a subdiagonal, and E(L) sits above the
!> diagonal in row L-1. A(1) and E(1) are unused; the generator poisons them.
!>
!> KL is the top of the convective boundary layer and is a runtime argument --
!> in CMAQ it varies per column, which is the part that makes this awkward to
!> vectorise. Rows above KL are untouched by the solver.
!>
!>   input:  nlays, nspcs, kl          (3 x int32)
!>           a(nlays)                  float32   first column, a(1) unused
!>           b(nlays)                  float32   diagonal
!>           e(nlays)                  float32   superdiagonal, e(1) unused
!>           d(nspcs, nlays)           float32   right-hand sides
!>   output: x(nspcs, nlays)           float32   solution; rows above kl are 0
program harness_matrix1

   use VGRD_DEFN, only: set_vgrid
   use CGRID_SPCS, only: set_species

   implicit none

   interface
      subroutine MATRIX1(KL, A, B, E, D, X)
         integer, intent(in) :: KL
         real, intent(in) :: A(:), B(:), E(:)
         real, intent(in) :: D(:, :)
         real, intent(out) :: X(:, :)
      end subroutine MATRIX1
   end interface

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios, nlays, nspcs, kl, k
   real, allocatable :: a(:), b(:), e(:), d(:, :), x(:, :), faces(:)

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_matrix1 <input.bin> <output.bin>'
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

   read (unit_in) nlays, nspcs, kl

   allocate (faces(nlays + 1))
   do k = 1, nlays + 1
      faces(k) = 1.0 - real(k - 1)/real(nlays)
   end do
   call set_vgrid(faces)
   call set_species(nspcs)

   allocate (a(nlays), b(nlays), e(nlays), d(nspcs, nlays), x(nspcs, nlays))
   read (unit_in) a
   read (unit_in) b
   read (unit_in) e
   read (unit_in) d
   close (unit_in)

   x = 0.0
   call MATRIX1(kl, a, b, e, d, x)

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) x
   close (unit_out)

end program harness_matrix1
