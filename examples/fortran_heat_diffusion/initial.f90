! EVOLVE-BLOCK-START
module heat_solver
    implicit none
contains
    subroutine initialize_state(n, state)
        integer, intent(in) :: n
        real(8), intent(out) :: state(n)
        integer :: i
        real(8) :: x

        do i = 1, n
            x = real(i - 1, 8) / real(n - 1, 8)
            state(i) = exp(-80.0d0 * (x - 0.35d0) * (x - 0.35d0))
            state(i) = state(i) + 0.5d0 * exp(-120.0d0 * (x - 0.72d0) * (x - 0.72d0))
        end do
        state(1) = 0.0d0
        state(n) = 0.0d0
    end subroutine initialize_state

    subroutine evolve_heat(n, steps, alpha, state)
        integer, intent(in) :: n
        integer, intent(in) :: steps
        real(8), intent(in) :: alpha
        real(8), intent(inout) :: state(n)
        real(8), allocatable :: next_state(:)
        integer :: step
        integer :: i

        allocate (next_state(n))
        do step = 1, steps
            next_state(1) = 0.0d0
            next_state(n) = 0.0d0
            do i = 2, n - 1
                next_state(i) = state(i) + alpha * &
                    (state(i - 1) - 2.0d0 * state(i) + state(i + 1))
            end do
            state = next_state
        end do
        deallocate (next_state)
    end subroutine evolve_heat

    real(8) function checksum(n, state)
        integer, intent(in) :: n
        real(8), intent(in) :: state(n)
        integer :: i

        checksum = 0.0d0
        do i = 1, n
            checksum = checksum + state(i) * real(i, 8)
        end do
    end function checksum

    subroutine solve_case(n, steps, alpha, answer)
        integer, intent(in) :: n
        integer, intent(in) :: steps
        real(8), intent(in) :: alpha
        real(8), intent(out) :: answer
        real(8), allocatable :: state(:)

        allocate (state(n))
        call initialize_state(n, state)
        call evolve_heat(n, steps, alpha, state)
        answer = checksum(n, state)
        deallocate (state)
    end subroutine solve_case
end module heat_solver
! EVOLVE-BLOCK-END


program main
    use heat_solver
    implicit none

    character(len=1024) :: input_path
    character(len=1024) :: output_path
    integer :: argc

    argc = command_argument_count()
    if (argc /= 2) then
        write (*, '(A)') "Usage: initial.f90 <input_path> <output_path>"
        stop 1
    end if

    call get_command_argument(1, input_path)
    call get_command_argument(2, output_path)
    call solve_file(trim(input_path), trim(output_path))

contains
    subroutine solve_file(input_file, output_file)
        character(len=*), intent(in) :: input_file
        character(len=*), intent(in) :: output_file
        integer :: in_unit
        integer :: out_unit
        integer :: status
        integer :: n
        integer :: steps
        real(8) :: alpha
        real(8) :: answer

        open (newunit=in_unit, file=input_file, status="old", action="read", iostat=status)
        if (status /= 0) then
            write (*, '(A)') "Could not open input file"
            stop 1
        end if

        open (newunit=out_unit, file=output_file, status="replace", action="write", iostat=status)
        if (status /= 0) then
            close (in_unit)
            write (*, '(A)') "Could not open output file"
            stop 1
        end if

        do
            read (in_unit, *, iostat=status) n, steps, alpha
            if (status /= 0) exit
            call solve_case(n, steps, alpha, answer)
            write (out_unit, '(ES24.16)') answer
        end do

        close (in_unit)
        close (out_unit)
    end subroutine solve_file
end program main
