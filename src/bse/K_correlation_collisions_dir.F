!
! License-Identifier: GPL
!
! Copyright (C) 2022 The Yambo Team
!
! Authors (see AUTHORS file for details): AM, DS
! 
subroutine K_correlation_collisions_dir(iq,i_block,qindx_ID_frag,Xk,q)
 !
 ! This routine evaluates the correlation collisions:
 !
 !  <n k1_bz |exp iG.r|m k2_bz> = <n k1_ibz s1 |exp iG.r|m k2_ibz s2>  =
 !
 use pars,          ONLY:SP
 use D_lattice,     ONLY:nsym
 use R_lattice,     ONLY:qindx_X,bz_samp
 use BS,            ONLY:O_ng,BS_bands,BS_res_K_corr,BS_blk,&
&                        BS_oscillators_alloc,K_CORR_collision,BS_T_grp,&
&                        BS_K_dim,l_BS_ares_from_res
 use electrons,     ONLY:n_sp_pol
 use collision_el,  ONLY:elemental_collision_free,elemental_collision_alloc
 use openmp,        ONLY:OPENMP_update,master_thread
 use timing_m,      ONLY:timing
 !
#include<dev_defs.h>
#include<memory.h>
 !
 integer,       intent(in)  :: iq,i_block,qindx_ID_frag
 type(bz_samp), intent(in)  :: Xk,q
 !
 ! Work Space
 !
 ! DEFINE BSK COMMON INDEXES
 !
 integer ::     i_k_bz,i_p_bz,i_k,i_p,i_kp_s,& 
&               i_kmq_bz,i_pmq_bz,i_kmq,i_pmq,i_pmq_s,i_kp_mq_s,i_p_s,i_kmq_s,i_k_s,& 
&               i_Tk,i_Tp,i_Tgrp_k,i_Tgrp_p,I_Tgrp_k_st,I_Tgrp_p_st,H_pos(2),& 
&               i_v_k,i_v_p,i_c_k,i_c_p,i_k_sp_pol_c,i_p_sp_pol_c,i_k_sp_pol_v,i_p_sp_pol_v,iq_W,& 
&               i_kmq_t,i_pmq_t 
 ! OMP LAST DEF
 integer ::     i_k_bz_last,i_p_bz_last,iq_W_bz_last,ig_W_last
 !
 integer     :: i_s_star,i_s_mq_star,i_s_collision,N_kmq,N_pmq
 character(1):: mode_now
 !
 logical, allocatable :: O_todo_table(:,:,:,:,:,:,:)
 !
 if (.not.BS_res_K_corr) return 
 !
 call timing('T_space CORR Osc.',OPR='start')
 !
 ! Every block is at k (ibz) fixed
 !
 i_Tgrp_k   =BS_blk(i_block)%iT_k
 i_Tgrp_p   =BS_blk(i_block)%iT_p
 i_Tgrp_k_st=BS_blk(i_block)%iT_k_st
 i_Tgrp_p_st=BS_blk(i_block)%iT_p_st
 i_k        =BS_blk(i_block)%ik
 i_p        =BS_blk(i_block)%ip
 !
 ! Table allocation
 !------------------
 YAMBO_ALLOC(BS_blk(i_block)%kp_table,(2,Xk%nibz))
 BS_blk(i_block)%kp_table=0
 !
 N_kmq=0
 N_pmq=0
 !
 do i_Tk=1,BS_T_grp(i_Tgrp_k)%size
   do i_Tp=1,BS_T_grp(i_Tgrp_p)%size
     !
     i_k_bz=BS_T_grp(i_Tgrp_k)%table(i_Tk,1) ! k
     i_p_bz=BS_T_grp(i_Tgrp_p)%table(i_Tp,1) ! p = k - q_W
     !
     i_kmq_bz=qindx_X(iq,i_k_bz,1) ! k-q
     i_pmq_bz=qindx_X(iq,i_p_bz,1) ! p-q = k - q_W -q
     !
     i_kmq   =Xk%sstar(i_kmq_bz,1)
     i_pmq   =Xk%sstar(i_pmq_bz,1)
     !
     i_kmq_t =BS_blk(i_block)%kp_table(1,i_kmq)
     i_pmq_t =BS_blk(i_block)%kp_table(2,i_pmq)     
     !
     if( i_kmq_t/=0 .and. i_pmq_t/=0 ) cycle
     !
     if( i_kmq_t==0 ) then
       N_kmq=N_kmq+1
       BS_blk(i_block)%kp_table(1,i_kmq)=N_kmq
     endif
     !
     if( i_pmq_t==0 ) then
       N_pmq=N_pmq+1
       BS_blk(i_block)%kp_table(2,i_pmq)=N_pmq
     endif
     !
   enddo
 enddo
 !
 YAMBO_ALLOC(BS_blk(i_block)%O_table1,(nsym,nsym,N_kmq,N_pmq,BS_bands(1):BS_bands(2),BS_bands(1):BS_bands(2),n_sp_pol))
 YAMBO_ALLOC(BS_blk(i_block)%O_table2,(nsym,nsym,N_kmq,N_pmq,BS_bands(1):BS_bands(2),BS_bands(1):BS_bands(2),n_sp_pol))
 allocate(O_todo_table(nsym,nsym,N_kmq,N_pmq,BS_bands(1):BS_bands(2),BS_bands(1):BS_bands(2),n_sp_pol))
 !
 BS_blk(i_block)%N_oscillators=0
 BS_blk(i_block)%O_table1=0
 BS_blk(i_block)%O_table2=0
 O_todo_table=.false.
 !
 i_p_bz_last=0 ;  i_k_bz_last=0 ;  iq_W_bz_last=0 ;  ig_W_last=0 
 !
 mode_now=BS_blk(i_block)%mode
 if (.not.l_BS_ares_from_res) mode_now="F"
 !
 select case (mode_now)
 case("R","A","F")
   !
   do i_Tk=1,BS_T_grp(i_Tgrp_k)%size
     do i_Tp=1,BS_T_grp(i_Tgrp_p)%size
       !
       H_pos(1) = sum(BS_T_grp(I_Tgrp_k_st:I_Tgrp_k-1)%size)+i_Tk+& 
&              (BS_T_grp(I_Tgrp_k)%i_res_ares-1)*BS_K_dim(1) 
       H_pos(2) = sum(BS_T_grp(I_Tgrp_p_st:I_Tgrp_p-1)%size)+i_Tp+& 
&              (BS_T_grp(I_Tgrp_p)%i_res_ares-1)*BS_K_dim(1) 
       !
       if (H_pos(1)>H_pos(2)) cycle
       !
       call fill_indexes(i_Tk,i_Tp,i_Tgrp_k,i_Tgrp_p,i_block,iq,               &
       &                        i_kp_s,i_s_star,i_kp_mq_s,i_s_mq_star,i_pmq_s, &
       &                        i_k,i_p,i_kmq,i_pmq,i_kmq_t,i_pmq_t,i_p_s,     &
       &                        i_c_p,i_v_p,i_p_sp_pol_v,i_p_sp_pol_c,i_kmq_s, &
       &                        i_v_k,i_c_k,i_k_sp_pol_v,i_k_sp_pol_c,i_k_s,   &
       &                        i_p_bz_last,i_k_bz_last,iq_W,iq_W_bz_last,ig_W_last)
       !
       if(i_k_sp_pol_c/=i_p_sp_pol_c) cycle
       !
       ! v->v', i.e. k-q -> p-q=k-q_W-q transitions
       !
       call EVAL_table(i_block,i_kmq_t,i_pmq_t,i_v_k,i_v_p,i_k_sp_pol_v,i_kmq_s,i_pmq_s)
       !
       ! c->c', i.e. k -> p=k-q_w transitions
       !
       call EVAL_table(i_block,1,1,i_c_k,i_c_p,i_k_sp_pol_c,i_k_s,i_p_s)
       !
     enddo
   enddo
   !
 case("C","Q")
   !
   do i_Tk=1,BS_T_grp(i_Tgrp_k)%size
     do i_Tp=1,BS_T_grp(i_Tgrp_p)%size
       !
       !
       H_pos(1) = sum(BS_T_grp(I_Tgrp_k_st:I_Tgrp_k-1)%size)+i_Tk+& 
&              (BS_T_grp(I_Tgrp_k)%i_res_ares-1)*BS_K_dim(1) 
       H_pos(2) = sum(BS_T_grp(I_Tgrp_p_st:I_Tgrp_p-1)%size)+i_Tp+& 
&              (BS_T_grp(I_Tgrp_p)%i_res_ares-1)*BS_K_dim(1) 
       !
       if (H_pos(1)>H_pos(2)) cycle
       !
       call fill_indexes(i_Tk,i_Tp,i_Tgrp_k,i_Tgrp_p,i_block,iq,               &
       &                        i_kp_s,i_s_star,i_kp_mq_s,i_s_mq_star,i_pmq_s, &
       &                        i_k,i_p,i_kmq,i_pmq,i_kmq_t,i_pmq_t,i_p_s,     &
       &                        i_c_p,i_v_p,i_p_sp_pol_v,i_p_sp_pol_c,i_kmq_s, &
       &                        i_v_k,i_c_k,i_k_sp_pol_v,i_k_sp_pol_c,i_k_s,   &
       &                        i_p_bz_last,i_k_bz_last,iq_W,iq_W_bz_last,ig_W_last)
       !
       if(i_k_sp_pol_c/=i_p_sp_pol_v) cycle
       !
       ! v->c', i.e. k-q -> p-q=k-q_W-q transitions
       !
       call EVAL_table(i_block,i_kmq_t,i_pmq_t,i_v_k,i_c_p,i_k_sp_pol_v,i_kmq_s,i_pmq_s)
       !
       ! c->v', i.e. k -> p=k-q_W transitions
       !
       call EVAL_table(i_block,1,1,i_c_k,i_v_p,i_k_sp_pol_c,i_k_s,i_p_s)
       !
     enddo
   enddo
   !
 end select
 !
 call BS_oscillators_alloc(.FALSE.,0,.TRUE.,i_block)
 !
#if !defined _CUDA
 !$omp parallel default(shared), private(i_Tk,i_Tp, &
 !$omp &          i_kp_s,i_s_star,H_pos,&
 !$omp &          i_kmq,i_pmq,i_kp_mq_s,i_pmq_s,i_s_mq_star,i_p_s,i_kmq_s,i_k_s,&
 !$omp &          i_v_k,i_v_p,i_c_k,i_c_p,i_k_sp_pol_c,i_p_sp_pol_c,i_k_sp_pol_v,i_p_sp_pol_v,iq_W,&
 !$omp &          i_kmq_t,i_pmq_t, &
 !$omp &          i_p_bz_last,i_k_bz_last,iq_W_bz_last,ig_W_last, &
 !$omp &          K_CORR_collision, i_s_collision)
#endif
 !
 call OPENMP_update(master_thread)
 !
 call elemental_collision_alloc(K_CORR_collision,NG=O_ng, TITLE="CORR_COLL")
 !
 i_p_bz_last=0 ;  i_k_bz_last=0 ;  iq_W_bz_last=0 ;  ig_W_last=0 
 !
 select case (mode_now)
 case("R","A","F")
   !
#if !defined _CUDA
   !$omp do collapse(2), schedule(dynamic)
#endif
   do i_Tk=1,BS_T_grp(i_Tgrp_k)%size
     do i_Tp=1,BS_T_grp(i_Tgrp_p)%size
       !
       H_pos(1) = sum(BS_T_grp(I_Tgrp_k_st:I_Tgrp_k-1)%size)+i_Tk+& 
&              (BS_T_grp(I_Tgrp_k)%i_res_ares-1)*BS_K_dim(1) 
       H_pos(2) = sum(BS_T_grp(I_Tgrp_p_st:I_Tgrp_p-1)%size)+i_Tp+& 
&              (BS_T_grp(I_Tgrp_p)%i_res_ares-1)*BS_K_dim(1) 
       !
       if (H_pos(1)>H_pos(2)) cycle
       !
       call fill_indexes(i_Tk,i_Tp,i_Tgrp_k,i_Tgrp_p,i_block,iq,               &
       &                        i_kp_s,i_s_star,i_kp_mq_s,i_s_mq_star,i_pmq_s, &
       &                        i_k,i_p,i_kmq,i_pmq,i_kmq_t,i_pmq_t,i_p_s,     &
       &                        i_c_p,i_v_p,i_p_sp_pol_v,i_p_sp_pol_c,i_kmq_s, &
       &                        i_v_k,i_c_k,i_k_sp_pol_v,i_k_sp_pol_c,i_k_s,   &
       &                        i_p_bz_last,i_k_bz_last,iq_W,iq_W_bz_last,ig_W_last)
       !
       if(i_k_sp_pol_c/=i_p_sp_pol_c) cycle
       !
       ! v->v', i.e. k-q -> p-q=k-q_W-q transitions
       !
       call EVAL_collision(i_block,i_kmq,i_pmq,i_kmq_t,i_pmq_t,i_k_sp_pol_v,i_p_sp_pol_v,&
       &                                   i_v_k,i_v_p,iq_W,i_kmq_s,i_pmq_s,K_CORR_collision)
       !
       ! c->c', i.e. k -> p=k-q_w transitions
       !
       call EVAL_collision(i_block,i_k,i_p,1,1,i_k_sp_pol_c,i_p_sp_pol_c,&
       &                        i_c_k,i_c_p,iq_W,i_k_s,i_p_s,K_CORR_collision)
       !
     enddo
   enddo
#if !defined _CUDA
   !$omp end do
#endif
   !
 case("C","Q")
   !
#if !defined _CUDA
   !$omp do collapse(2), schedule(dynamic)
#endif
   do i_Tk=1,BS_T_grp(i_Tgrp_k)%size
     do i_Tp=1,BS_T_grp(i_Tgrp_p)%size
       !
       H_pos(1) = sum(BS_T_grp(I_Tgrp_k_st:I_Tgrp_k-1)%size)+i_Tk+& 
&              (BS_T_grp(I_Tgrp_k)%i_res_ares-1)*BS_K_dim(1) 
       H_pos(2) = sum(BS_T_grp(I_Tgrp_p_st:I_Tgrp_p-1)%size)+i_Tp+& 
&              (BS_T_grp(I_Tgrp_p)%i_res_ares-1)*BS_K_dim(1) 
       !
       if (H_pos(1)>H_pos(2)) cycle
       !
       call fill_indexes(i_Tk,i_Tp,i_Tgrp_k,i_Tgrp_p,i_block,iq,               &
       &                        i_kp_s,i_s_star,i_kp_mq_s,i_s_mq_star,i_pmq_s, &
       &                        i_k,i_p,i_kmq,i_pmq,i_kmq_t,i_pmq_t,i_p_s,     &
       &                        i_c_p,i_v_p,i_p_sp_pol_v,i_p_sp_pol_c,i_kmq_s, &
       &                        i_v_k,i_c_k,i_k_sp_pol_v,i_k_sp_pol_c,i_k_s,   &
       &                        i_p_bz_last,i_k_bz_last,iq_W,iq_W_bz_last,ig_W_last)
       !
       if(i_k_sp_pol_c/=i_p_sp_pol_v) cycle
       !
       ! v->c', i.e. k-q -> p-q=k-q_W-q transitions
       !
       call EVAL_collision(i_block,i_kmq,i_pmq,i_kmq_t,i_pmq_t,i_k_sp_pol_v,&
       &                        i_p_sp_pol_c,i_v_k,i_c_p,iq_W,i_kmq_s,i_pmq_s,K_CORR_collision)
       !
       ! c->v', i.e. k -> p=k-q_W transitions
       !
       call EVAL_collision(i_block,i_k,i_p,1,1,i_k_sp_pol_c,i_p_sp_pol_v,&
       &                        i_c_k,i_v_p,iq_W,i_k_s,i_p_s,K_CORR_collision)
       !
     enddo
   enddo
   !
#if !defined _CUDA
   !$omp end do
#endif
   !
 end select
 !
 call elemental_collision_free(K_CORR_collision)
#if !defined _CUDA
 !$omp end parallel
#endif
 !
 call timing('T_space CORR Osc.',OPR='stop')
 !
 contains
   !
   subroutine fill_indexes(j_Tk,j_Tp,j_Tgrp_k,j_Tgrp_p,j_block,j_q,            &
       &                        j_kp_s,j_s_star,j_kp_mq_s,j_s_mq_star,j_pmq_s, &
       &                        j_k,j_p,j_kmq,j_pmq,j_kmq_t,j_pmq_t,j_p_s,     &
       &                        j_c_p,j_v_p,j_p_sp_pol_v,j_p_sp_pol_c,j_kmq_s, &
       &                        j_v_k,j_c_k,j_k_sp_pol_v,j_k_sp_pol_c,j_k_s,   &
       &                        j_p_bz_last,j_k_bz_last,j_q_W,j_q_W_bz_last,j_g_W_last)
       !
       use R_lattice,     ONLY:qindx_B,qindx_B_load
       use D_lattice,     ONLY:sop_inv,sop_tab
       use parallel_m,    ONLY:PAR_K_scheme
       !
       implicit none
       !
       integer, intent(in)   :: j_Tk,j_Tp,j_Tgrp_k,j_Tgrp_p,j_block,j_q
       integer, intent(out)  :: j_kp_s,j_s_star,j_kp_mq_s,j_s_mq_star,j_pmq_s,  &
       &                        j_k,j_p,j_kmq,j_pmq,j_kmq_t,j_pmq_t,j_p_s,      &
       &                        j_c_p,j_v_p,j_p_sp_pol_v,j_p_sp_pol_c,j_kmq_s,  &
       &                        j_v_k,j_c_k,j_k_sp_pol_v,j_k_sp_pol_c,j_q_W,j_k_s
       integer, intent(inout):: j_p_bz_last,j_k_bz_last,j_q_W_bz_last,j_g_W_last
       !
       ! local variables
       !
       integer               :: j_k_bz,j_p_bz,j_kmq_bz,j_pmq_bz,j_k_s_m1,  &
       &                        j_kmq_s_m1,j_k_bz_mem,qindx_tmp(2),    &
       &                        j_q_W_bz,j_q_W_s,j_g_W
       !
       j_k_bz  = BS_T_grp(j_Tgrp_k)%table(j_Tk,1) 
       j_p_bz  = BS_T_grp(j_Tgrp_p)%table(j_Tp,1) 
       ! 
       j_k_s =Xk%sstar(j_k_bz,2) 
       j_p_s =Xk%sstar(j_p_bz,2) 
       ! 
       j_k_s_m1= sop_inv(j_k_s) 
       j_kp_s  = sop_tab(j_k_s_m1,j_p_s) 
       ! 
       j_kmq_bz=qindx_X(j_q,j_k_bz,1) 
       j_pmq_bz=qindx_X(j_q,j_p_bz,1) 
       ! 
       j_kmq   =Xk%sstar(j_kmq_bz,1) 
       j_pmq   =Xk%sstar(j_pmq_bz,1) 
       ! 
       j_kmq_s =Xk%sstar(j_kmq_bz,2) 
       j_pmq_s =Xk%sstar(j_pmq_bz,2) 
       ! 
       j_kmq_t=BS_blk(j_block)%kp_table(1,j_kmq) 
       j_pmq_t=BS_blk(j_block)%kp_table(2,j_pmq) 
       !
       j_kmq_s_m1 = sop_inv(j_kmq_s) 
       j_kp_mq_s  = sop_tab(j_kmq_s_m1,j_pmq_s) 
       ! 
       j_k_bz_mem=PAR_K_scheme%bz_index(j_k_bz) 
       if (j_p_bz_last/=j_p_bz.or.j_k_bz_last/=j_k_bz) then 
         j_p_bz_last=j_p_bz 
         j_k_bz_last=j_k_bz 
         if (j_k_bz_mem==0) then 
           !$omp critical 
           qindx_tmp=qindx_B_load(j_p_bz,j_k_bz,qindx_ID_frag) 
           j_q_W_bz=qindx_tmp(1) 
           j_g_W   =qindx_tmp(2) 
           !$omp end critical 
         else 
           j_q_W_bz=qindx_B(j_p_bz,j_k_bz_mem,1) 
           j_g_W   =qindx_B(j_p_bz,j_k_bz_mem,2) 
         endif 
         j_q_W_bz_last=j_q_W_bz 
         j_g_W_last   =j_g_W    
       else 
         j_q_W_bz=j_q_W_bz_last 
         j_g_W   =j_g_W_last    
       endif 
       j_q_W   =q%sstar( j_q_W_bz ,1) 
       j_q_W_s =q%sstar( j_q_W_bz ,2) 
       ! 
       j_v_k=BS_T_grp(j_Tgrp_k)%table(j_Tk,2) 
       j_c_k=BS_T_grp(j_Tgrp_k)%table(j_Tk,3) 
       ! 
       j_v_p=BS_T_grp(j_Tgrp_p)%table(j_Tp,2) 
       j_c_p=BS_T_grp(j_Tgrp_p)%table(j_Tp,3) 
       ! 
       j_k_sp_pol_c=BS_T_grp(j_Tgrp_k)%table(j_Tk,4) 
       j_p_sp_pol_c=BS_T_grp(j_Tgrp_p)%table(j_Tp,4) 
       j_k_sp_pol_v=BS_T_grp(j_Tgrp_k)%table(j_Tk,5) 
       j_p_sp_pol_v=BS_T_grp(j_Tgrp_p)%table(j_Tp,5) 
       ! 
       j_s_star=Xk%s_table(j_p,j_kp_s) 
       j_s_mq_star=Xk%s_table(j_pmq,j_kp_mq_s) 
       !
   end subroutine fill_indexes
   !
   subroutine EVAL_table(iB,iki,iko,ibi,ibo,i_sp_pol,i_s_ki,i_s_ko)
     implicit none
     integer, intent(in) :: iB,iki,iko,ibi,ibo,i_sp_pol,i_s_ki,i_s_ko
     !
     if( O_todo_table(i_s_ki,i_s_ko,iki,iko,ibi,ibo,i_sp_pol) ) return
     !
     BS_blk(iB)%N_oscillators=BS_blk(iB)%N_oscillators+1 
     BS_blk(iB)%O_table1(i_s_ki,i_s_ko,iki,iko,ibi,ibo,i_sp_pol)=BS_blk(iB)%N_oscillators
     BS_blk(iB)%O_table2(i_s_ki,i_s_ko,iki,iko,ibi,ibo,i_sp_pol)=iB
     O_todo_table(i_s_ki,i_s_ko,iki,iko,ibi,ibo,i_sp_pol)=.true.
     !
   end subroutine
   !
   subroutine EVAL_collision(iB,iki,iko,ikit,ikot,i_sp_pol_i,i_sp_pol_o,ibi,ibo,iq_W_in,i_s_ki,i_s_ko,isc)
     use collision_el,  ONLY:elemental_collision
     use deviceXlib_m,  ONLY:dev_memcpy
     implicit none
     integer,intent(in) :: iB,iki,iko,ikit,ikot,ibi,ibo,i_sp_pol_i,i_sp_pol_o,iq_W_in,i_s_ki,i_s_ko
     type(elemental_collision),intent(inout) :: isc
     integer :: iB_oscillator
     if (.not.O_todo_table(i_s_ki,i_s_ko,ikit,ikot,ibi,ibo,i_sp_pol_i)) return
     isc%is=(/ibi,iki,i_s_ki,i_sp_pol_i/)
     isc%os=(/ibo,iko,i_s_ko,i_sp_pol_o/)
     isc%qs=(/1,iq_W_in,1/)
     !
     iB_oscillator = BS_blk(iB)%O_table1(i_s_ki,i_s_ko,ikit,ikot,ibi,ibo,i_sp_pol_i)
     !
     call DEV_SUB(scatter_Bamp)(isc)
     call dev_memcpy( DEV_VAR(BS_blk(iB)%O_c)(:,iB_oscillator), DEV_VAR(isc%rhotw) )
     !
     O_todo_table(i_s_ki,i_s_ko,ikit,ikot,ibi,ibo,i_sp_pol_i) = .false.
   end subroutine
   !
end subroutine K_correlation_collisions_dir
