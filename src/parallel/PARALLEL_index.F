!
! License-Identifier: GPL
!
! Copyright (C) 2013 The Yambo Team
!
! Authors (see AUTHORS file for details): AM DS
!
subroutine PARALLEL_index(px,uplim,low_range,COMM,CONSECUTIVE,ORDERED,NO_EMPTIES,MASK)
 !
 use pars,        ONLY:SP
 use parallel_m,  ONLY:PP_indexes,ncpu,myid,p_prod,mpi_comm_world,&
&                      yMPI_comm,comm_default_value,PP_indexes_reset
 use parallel_int,ONLY:PP_redux_wait
 use stderr,      ONLY:intc
 use vec_operate, ONLY:sort
 !
#include<memory.h>
 !
 type(PP_indexes)       ::px
 integer                ::uplim(:)
 integer,       optional::low_range(:)
 type(yMPI_comm),optional::COMM
 logical       ,optional::CONSECUTIVE
 logical       ,optional::ORDERED
 logical       ,optional::NO_EMPTIES
 logical       ,optional::MASK(:)
 ! 
 ! Work Space
 !
 integer :: nindxs,i1,i2,i3,d2,ic,average_steps,tot_steps,myid_loop,ncpu_loop,local_comm,&
&           i_range(2),missing_cpus,par_kind_2D
 logical :: NO_EMPTIES_,SERIAL,local_ORDERED,local_CONSECUTIVE
 integer, allocatable :: i_done(:),lowlim(:),W_out_index(:),W_in_index(:)
 real(SP),allocatable :: W_out(:),W_out_sorted(:),W_in(:)
 !
 myid_loop =myid
 ncpu_loop =ncpu
 local_comm=mpi_comm_world
 SERIAL    =.FALSE.
 !
 if (present(COMM)) then
   ncpu_loop =COMM%n_CPU
   myid_loop =COMM%CPU_id
   local_comm=COMM%COMM
 endif
 !
 local_ORDERED=.false.
 if(present(ORDERED)) local_ORDERED=ORDERED
 !
 local_CONSECUTIVE=.false.
 if(present(CONSECUTIVE)) local_CONSECUTIVE=CONSECUTIVE
 !
 if (ncpu_loop==1) then
   myid_loop =0
   local_comm=comm_default_value
   SERIAL    =.TRUE.
 endif
 !
 nindxs=size(uplim)
 allocate(lowlim(nindxs))
 lowlim=1
 if (present(low_range))  lowlim=low_range
 !
 if (allocated(px%n_of_elements) .or. allocated(px%element_1D) .or. allocated(px%element_2D) ) then 
   call PP_indexes_reset(px)
 endif
 !
 YAMBO_ALLOC(px%n_of_elements,(ncpu_loop))
 px%n_of_elements=0
 !
 if (uplim(1)==0) goto 1
 !
 if (nindxs==1.and.local_CONSECUTIVE) then
   !
   YAMBO_ALLOC(px%element_1D,(uplim(1)))
   YAMBO_ALLOC(px%first_of_1D,(ncpu_loop))
   YAMBO_ALLOC(px%last_of_1D,(ncpu_loop))
   allocate(i_done(uplim(1)))
   px%element_1D=.false.
   px%first_of_1D=0
   px%last_of_1D=0
   i_done=1
   i_done(:lowlim(1))=0
   average_steps=(uplim(1)-lowlim(1)+1)/ncpu_loop
   missing_cpus=(uplim(1)-lowlim(1)+1)-average_steps*ncpu_loop
   do i1=1,ncpu_loop
     if (i1==myid_loop+1) then
       px%n_of_elements(myid_loop+1)=average_steps
       i_range=(/lowlim(1)+average_steps*(i1-1),lowlim(1)+average_steps*i1-1/)
       if(myid_loop+1<=missing_cpus) then
         i_range(1)=i_range(1)+myid_loop
         i_range(2)=i_range(2)+myid_loop+1
         px%n_of_elements(myid_loop+1)=px%n_of_elements(myid_loop+1)+1
       else
         i_range(1)=i_range(1)+missing_cpus
         i_range(2)=i_range(2)+missing_cpus
       endif
       px%first_of_1D(myid_loop+1)=i_range(1)
       px%last_of_1D(myid_loop+1)=i_range(2)
       px%element_1D(i_range(1):i_range(2))=.TRUE.
       i_done(i_range(1):i_range(2))=i_done(i_range(1):i_range(2))-1
     endif
   enddo
   !
   if (.not.SERIAL) call PP_redux_wait(i_done,p_prod,COMM=local_comm)
   if (any(i_done==1)) call error('Incomplete Parallel Index Filling')
   !
 else if (nindxs==1.and..not.local_CONSECUTIVE) then
   !
   if (.not.allocated(px%weight_1D)) then
     YAMBO_ALLOC(px%weight_1D,(uplim(1)))
     px%weight_1D=1
     if (lowlim(1)>1) px%weight_1D(1:lowlim(1)-1)=0
   endif
   !
   allocate(i_done(uplim(1)))
   YAMBO_ALLOC(px%element_1D,(uplim(1)))
   YAMBO_ALLOC(px%first_of_1D,(ncpu_loop))
   px%first_of_1D(myid_loop+1)=-1
   px%element_1D=.false.
   tot_steps=sum(px%weight_1D)
   i_done=1
   if (lowlim(1)>1) i_done(:lowlim(1)-1) =0
   average_steps=tot_steps/ncpu_loop
   if( mod(tot_steps,ncpu_loop)>0 ) average_steps=average_steps+1
   !
   ! Here the linear chain is distributed
   !
   allocate(W_in(uplim(1)-lowlim(1)+1),W_in_index(uplim(1)-lowlim(1)+1))
   do i1=lowlim(1),uplim(1)
     W_in(i1-lowlim(1)+1)=px%weight_1D(i1)
   enddo
   call sort(arrin=W_in,indx=W_in_index)
   allocate(W_out(ncpu_loop),W_out_sorted(ncpu_loop),W_out_index(ncpu_loop))
   W_out =0._SP
   do i1=uplim(1)-lowlim(1)+1,1,-1
     call sort(arrin=W_out,arrout=W_out_sorted,indx=W_out_index)
     W_out(W_out_index(1))=W_out(W_out_index(1))+W_in(i1)
     if (myid_loop==W_out_index(1)-1) px%element_1D( W_in_index(i1) )=.true.
   enddo
   !
   do i1=lowlim(1),uplim(1)
     if (.not.px%element_1D(i1)) cycle
     if (px%first_of_1D(myid_loop+1)==-1) px%first_of_1D(myid_loop+1)=i1
     i_done(i1)=0
     px%n_of_elements(myid_loop+1)=px%n_of_elements(myid_loop+1)+px%weight_1D(i1)
   enddo
   if (.not.SERIAL) call PP_redux_wait(i_done,p_prod,COMM=local_comm)
   !
   ! Fill the remaining elements trying to not go beyond the average_steps
   !
   do ic=1,ncpu_loop
     if (ic==myid_loop+1) then
       do i1=lowlim(1),uplim(1)
         if (i_done(i1)==0) cycle
         if (px%n_of_elements(myid_loop+1)>=average_steps) cycle
         px%element_1D(i1)=.true.
         i_done(i1)=0
         px%n_of_elements(myid_loop+1)=px%n_of_elements(myid_loop+1)+px%weight_1D(i1)
       enddo
     endif
     if (.not.SERIAL) call PP_redux_wait(i_done,p_prod,COMM=local_comm)
   enddo
   !
   if (any(i_done==1)) call error('Incomplete Parallel Index Filling')
   !
   deallocate(W_in,W_in_index,W_out,W_out_sorted,W_out_index)
   !
 else if (nindxs==2) then
   !
   if (     local_ORDERED.and..not.present(MASK)) par_kind_2D=1
   if (     local_ORDERED.and.     present(MASK)) par_kind_2D=2
   if (.not.local_ORDERED.and..not.present(MASK)) par_kind_2D=3
   if (.not.local_ORDERED.and.     present(MASK)) par_kind_2D=4
   !
   if (par_kind_2D==1) call error(' PARALLEL index structure not implemented')
   !
   YAMBO_ALLOC(px%element_2D,(uplim(1),uplim(2)))
   px%element_2D=.false.
   !
   if (present(MASK)) then
     !
     i3=0
     !
     do i1=lowlim(1),uplim(1)
       do i2=lowlim(2),uplim(2)
         !
         if ( local_ORDERED .and. i2<i1 ) cycle
         if ( .not.( (MASK(i1).and.abs(i2-i1)<=uplim(1)/2) .or. (MASK(i2).and.abs(i2-i1)>uplim(1)/2) ) ) cycle
         !
         i3=i3+1
         if ( .not.(mod(i3+myid_loop,ncpu_loop)==0) ) cycle
         !
         px%element_2D(i1,i2)=.true.
         px%n_of_elements(myid_loop+1)=px%n_of_elements(myid_loop+1)+1
         !
       enddo
     enddo
     !
   else
     !
     d2=uplim(2)-lowlim(2)+1
     !
     do i1=lowlim(1),uplim(1)
       do i2=lowlim(2),uplim(2)
         i3=(i1-lowlim(1))*d2+i2-lowlim(2)+1-myid
         if (mod(i3,ncpu)==0) px%element_2D(i1,i2)=.true.
         if (px%element_2D(i1,i2)) px%n_of_elements(myid+1)=px%n_of_elements(myid+1)+1
       enddo
     enddo
     !
   endif
   !
 endif
 !
1 deallocate(lowlim)
 if(allocated(i_done)) deallocate(i_done)
 !
 NO_EMPTIES_=.false.
 if (present(NO_EMPTIES)) NO_EMPTIES_=NO_EMPTIES
 !
 if (px%n_of_elements(myid_loop+1)==0) then
   call warning("Empty workload for CPU "//trim(intc(myid+1)))
 endif
 !
end subroutine
