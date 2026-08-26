!> Golden harness for ZFDBC -- CMAQ's zero-flux-divergence outflow boundary
!> condition (zfdbc.f, after Pleim, JGR 1991).
!>
!>   usage: harness_zfdbc <input.bin> <output.bin>
!>
!> Unlike the PPM kernels this one is stateless -- no SAVEd arrays, no first-call
!> sizing -- so a single process can evaluate every case in one pass.
!>
!>   input:  ncase                      (int32)
!>           c1(ncase), c2(ncase)       float32   near and next-nearest cell
!>           v1(ncase), v2(ncase)       float32   nearest and next face velocity
!>   output: result(ncase)              float32
program harness_zfdbc

   implicit none

   real, external :: ZFDBC

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios, i, ncase
   real, allocatable :: c1(:), c2(:), v1(:), v2(:), result(:)

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_zfdbc <input.bin> <output.bin>'
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

   read (unit_in) ncase
   allocate (c1(ncase), c2(ncase), v1(ncase), v2(ncase), result(ncase))
   read (unit_in) c1
   read (unit_in) c2
   read (unit_in) v1
   read (unit_in) v2
   close (unit_in)

   do i = 1, ncase
      result(i) = ZFDBC(c1(i), c2(i), v1(i), v2(i))
   end do

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) result
   close (unit_out)

end program harness_zfdbc
