!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): AM
!
subroutine QP_driver(X,Xen,Xk,en,k,q,Xw,Dip)
 !
 use pars,          ONLY:SP,pi,cONE,cZERO
 use units,         ONLY:HA2EV
 use drivers,       ONLY:l_life,l_ppa,l_mpa,l_elel_corr,l_elph_corr,l_cohsex,&
&                        Finite_Tel,l_elphoton_corr,l_rim_w
 use electrons,     ONLY:levels,n_sp_pol,spin
 use R_lattice,     ONLY:bz_samp,RIM_W_is_diagonal,RIM_ng,RIM_W_ng,RIM_W_for_graph
 use DIPOLES,       ONLY:DIPOLE_t
 use frequency,     ONLY:w_samp
 use com,           ONLY:msg
 use X_m,           ONLY:X_t
 use QP_m,          ONLY:QP_t,QP_dSc,QP_dSc_steps,QP_Vxc,QP_Sc,QP_solver,QP_table,QP_n_G_bands,&
&                        QP_Vnl_xc,QP_n_states,QP_nb,QP_G_zoom_er,QP_G_Zoom_treshold,&
&                        QP_nk,QP_state,QP_dSc_test,QP_reset,use_GreenF_Zoom,l_QP_symmetrize,&
&                        QP_Sc_steps,QP_G_er,QP_G_dr,QP_SET_temperature_pre_factor,&
&                        GF_is_causal,QP_G_amplitude_integral,use_GreenF_to_eval_QP,&
&                        l_GW_terminator,QP_G_info
 use parser_m,      ONLY:parser
#if defined _ELPH
 use ELPH,          ONLY:QP_DW,QP_OMS_Fan
#endif
 use IO_int,        ONLY:io_control
 use IO_m,          ONLY:OP_RD_CL,VERIFY,REP,NONE,IO_NO_BINDING_ERROR
 use global_XC,     ONLY:QP_DB_kind,QP_SE_COHSEX,QP_SE_GoWo_PPA,QP_SE_GoWo,QP_SE_GWo_PPA,&
&                        QP_SE_GWo,QP_SE_POLARON,QP_SE_GW_QED,QP_SE_GW_PPA
 use interfaces,    ONLY:eval_G_minus_G
 use parallel_m,    ONLY:PAR_COM_WORLD,PAR_IND_WF_linear
 use stderr,        ONLY:intc
 use descriptors,   ONLY:IO_desc,IO_desc_reset,IO_desc_add
 !
#include<memory.h>
 !
 type(levels)  ::en,Xen
 type(bz_samp) ::Xk,k,q
 type(X_t)     ::X
 type(w_samp)  ::Xw
 type(DIPOLE_t)::Dip
 !
 ! Work Space
 !
 type(QP_t)        ::qp
 integer           ::i1,io_QP_and_GF_err,io_G_err,ID_QP,ID_G,i_spin
 integer           ::lw_bnd
 integer, external ::io_QP_and_GF
 real(SP),external ::G2E
 ! frequency grid for Sc in the Green Function
 real(SP),allocatable ::W(:)
 !
 ! Logicals
 !==========
 call QP_logicals( )
 !
 ! Some check
 if (QP_n_G_bands(2)>en%nb) call error("Too many bands required in Green's function. Run DFT with more bands!")
 !
 if (l_QP_symmetrize) call warning("Symmetrize states with the same energy: in rare case this symmetrization can introduce errors")
 !
 call parser('RIM_W_diag',RIM_W_is_diagonal)
 call parser('RIM_W_graph',RIM_W_for_graph)
 if(l_rim_w .and. RIM_W_ng == 0) RIM_W_ng = RIM_ng
 !
 ! The non-local (Vnl) and local (Vxc) part of the xc potentials are not always needed
 ! 
 ! They can be not necessary in lifetimes calculations or for SC calculations.
 !
 !======================================================================================
#if !defined _SC
 if (.not.allocated(QP_Vnl_xc).and..not.allocated(QP_Vxc).and.l_elel_corr.and..not.l_life) return
#endif
 !
 if (.not.any((/l_elel_corr,l_elph_corr,l_elphoton_corr/))) then
   call warning('e-p, e-e and e-photon correlation switched off')
   return
 endif
 !
 ! Head message
 !==============
 call QP_reset(qp)
 !
 ! Setup Temperature pre-factor
 !==============================
 call QP_SET_temperature_pre_factor( )
 !
 call IO_desc_reset(qp%desc)
 !
 if (trim(QP_solver)=='n') then
   !
   call section('*','Dyson equation: Newton solver')
   !
   call IO_desc_add(qp%desc,kind="s",str='GW solver',S="Newton",check=.TRUE.,ops=(/"=="/))
   !
 else if (trim(QP_solver)=='s') then
   !
   call section('*','Dyson equation: non perturbative QP_secant method')
   !
   call IO_desc_add(qp%desc,kind="s",str='GW solver',S="Secant",check=.TRUE.,ops=(/"=="/))
   !
 else if (trim(QP_solver)=='g') then
   call section('*','Dyson equation: full Green`s function')
   call msg('r', '[Green] Sc steps                                         ',QP_Sc_steps)
   call msg('r', '[Green] Sc energy range (centered in the bare value)     ',QP_G_er*HA2EV,"[eV]")
   call msg('rn','[Green] Sc damping range                                 ',QP_G_dr*HA2EV,"[eV]")
   !
   call IO_desc_add(qp%desc,kind="s",str='GW solver',S="Full Green`s function",check=.TRUE.,ops=(/"=="/))
   !
   GF_is_causal=Finite_Tel.or.l_elph_corr 
   !
 else if (l_life) then
   call section('*','Dyson equation: QP lifetimes')
   !
 else
   return
   !
 endif
 !
 if (l_ppa) then
   !
   call IO_desc_add(qp%desc,kind="s",str='GW approximation',S="PPA",check=.TRUE.,ops=(/"=="/))
   call IO_desc_add(qp%desc,kind="r",str='PPA imaginary Energy',R=(/X%ppaE*HA2EV/),check=.TRUE.,ops=(/"=="/),term="ev")
   !
   QP_DB_kind=QP_SE_GoWo_PPA
   !
 else if (l_mpa) then
   !
   call IO_desc_add(qp%desc,kind="s",str='GW approximation',S="MPA",check=.TRUE.,ops=(/"=="/))
   call IO_desc_add(qp%desc,kind="r",str='MPA number of poles',I=(/X%mpa_npoles/),check=.TRUE.,ops=(/"=="/))
   !
   QP_DB_kind=QP_SE_GoWo_PPA
   !
 else if (l_cohsex) then
   !
   call IO_desc_add(qp%desc,kind="s",str='GW approximation',S="COHSEX",check=.TRUE.,ops=(/"=="/))
   !
   QP_DB_kind=QP_SE_COHSEX
   !
 else
   !
   if (l_elel_corr) then
      call IO_desc_add(qp%desc,kind="s",str='GW approximation',S="Real Axis",check=.TRUE.,ops=(/"=="/))
   else if (l_elph_corr) then
      call IO_desc_add(qp%desc,kind="s",str='GW approximation',S="Fan + DW",check=.TRUE.,ops=(/"=="/))
   else if (l_elphoton_corr) then
      call IO_desc_add(qp%desc,kind="s",str='GW approximation',S="Lamb",check=.TRUE.,ops=(/"=="/))
   endif
   !
   ! Here I am not considering the case where ELPH + GW is used.
   ! For this case I need to create a new global KIND.
   !
   QP_DB_kind=QP_SE_GoWo
   !
   if (l_elph_corr) QP_DB_kind=QP_SE_POLARON 
   !
   if (l_elphoton_corr) QP_DB_kind=QP_SE_GW_QED
   !
 endif
 !
 ! Basic defs
 !============
 !
 call QP_state_table_setup(en)
 !
 ! Here I copy several informations to the qp_t type.
 ! This is because I want the qp to be completely independent of the run
 ! in order to be possibly read as QP correction.
 !
 qp%nk=QP_nk
 qp%nb=QP_nb
 qp%n_states=QP_n_states
 !
 ! In lifetimes calculations  the X db may be not
 ! present. So I need to define some variables that 
 ! must be correctly written in the QP%desc(s)
 !
 if (l_life) then
   call X_pre_setup(Xen,Xk,X)
   if (X%ng_db==0) X%ng_db=X%ng
 endif
 !
 if (l_ppa.and.l_GW_terminator) then
   X%ng=eval_G_minus_G(X%ng,0,COMM=PAR_COM_WORLD)
   call warning(' Response block size in GW reduced to '//intc(X%ng)//'RL ('//intc(int(1000.*G2E(X%ng)))//'mHa)')
 endif
 !
 ! Local Allocations
 !===================
 !
 call local_alloc()
 !
 call QP_descriptions(qp,X,Xw,.FALSE.)
 !
 do i1=1,QP_n_states
   !
   i_spin=spin(QP_table(i1,:))
   !
   ! To perform the SC GWo the bare interaction are needed. Those are
   ! stored in the %E array, or in %Eo array if an initial QP correction
   ! has been added already
   !
   qp%E(i1)=en%E(QP_table(i1,1),QP_table(i1,3),i_spin)
   if (allocated(en%Eo)) then
      qp%E_bare(i1)=en%Eo(QP_table(i1,1),QP_table(i1,3),i_spin)
   else
      qp%E_bare(i1)=qp%E(i1)
   endif
   !
   qp%k(QP_table(i1,3),:)=k%pt(QP_table(i1,3),:)
   !
 enddo
 !
 if (.not.l_life) then
   !
   ! QP I/O
   !========
   !
   call io_control(ACTION=OP_RD_CL,COM=REP,SEC=(/1,2/),MODE=VERIFY,ID=ID_QP)
   io_QP_and_GF_err=io_QP_and_GF('QP',qp,ID_QP)
   !
   if (io_QP_and_GF_err==0.and..not.trim(QP_solver)=='g') then
     call io_control(ACTION=OP_RD_CL,COM=NONE,SEC=(/1,2,3/),ID=ID_QP)
     io_QP_and_GF_err=io_QP_and_GF('QP',qp,ID_QP)
     call QP_report_and_write(k,qp,en,0)
     call local_free()
     return
   endif
   !
   ! Green Functions I/O
   !=====================
   !
   if (trim(QP_solver)=='g') then
     !
     call io_control(ACTION=OP_RD_CL,COM=REP,SEC=(/1,2/),MODE=VERIFY,ID=ID_G)
     io_G_err=io_QP_and_GF('G',qp,ID_G)
     !
     if (io_G_err==0.or.io_G_err==IO_NO_BINDING_ERROR) then
       !
       if (use_GreenF_to_eval_QP) use_GreenF_to_eval_QP=(.not.io_QP_and_GF_err==0).and.(io_G_err==0)
       if (use_GreenF_Zoom)       use_GreenF_Zoom=io_G_err==IO_NO_BINDING_ERROR.and..not.use_GreenF_to_eval_QP
       !
       call io_control(ACTION=OP_RD_CL,COM=NONE,SEC=(/1,2,3/),ID=ID_G)
       io_G_err=io_QP_and_GF('G',qp,ID_G)
       !
       if (use_GreenF_to_eval_QP) then
         call QP_Green_Function(qp,en,0) 
         call QP_report_and_write(k,qp,en,-1)
         call local_free()
         return
       endif
       !
       if (use_GreenF_Zoom) then
         YAMBO_ALLOC(QP_G_zoom_er,(QP_n_states,2))
       else
         QP_G_Zoom_treshold=0._SP
       endif
       call QP_Green_Function(qp,en,0) 
       if (.not.use_GreenF_Zoom) then
         call QP_report_and_write(k,qp,en,0)
         call local_free()
         return
       endif
       !
     else
       QP_G_Zoom_treshold=0._SP
     endif
     !
   endif
   !
 endif
 !
 ! Updated descriptions (needed in case of Zoomed GFs)
 !=====================================================
 !
 call QP_descriptions(qp,X,Xw,.TRUE.)
 !
 ! Solvers 
 !=========
 !
 if (trim(QP_solver)=='n') then
   !
   ! NEWTON
   !--------
   call QP_newton(X,Xen,Xk,en,k,q,qp,Xw,Dip)
   !
 endif
 !
 if (trim(QP_solver)=='s') then
   !
   ! SECANT
   !--------
   call QP_secant_driver(X,Xen,Xk,en,k,q,qp,Xw)
   !
 endif
 !
 ! GREEN`s FUNCTIONS
 !-------------------
 !
 if (trim(QP_solver)=='g') then
   !
   if (l_elel_corr.and.(.not.l_ppa.and..not.l_mpa.and..not.l_cohsex)) &
&                 call QP_real_axis(X,Xen,Xk,en,k,q,qp,Xw,Dip,0)
   !
   ! DALV: GF for ppa and mpa
   !
   if (l_elel_corr.and.(l_ppa.or.l_mpa)) then
     !
     YAMBO_ALLOC(W,(QP_Sc_steps))
     !
     QP_dSc_steps=QP_Sc_steps
     W(1)=QP_G_er(1)
     !
     forall(i1=1:QP_Sc_steps-1) W(i1+1)=W(1)+(QP_G_er(2)-QP_G_er(1))*real(i1,SP)/(QP_Sc_steps-1._SP)
     !
     if ( l_ppa ) call QP_ppa_cohsex(X,Xk,en,k,q,qp,Xw,W,0)
     if ( l_mpa ) call QP_mpa(X,Xk,en,k,q,qp,Xw,W,0)
     !
     YAMBO_FREE(W)
     !
   endif
   !
#if defined _ELPH 
   !
   if (l_elph_corr) call ELPH_Sigma_c(en,k,q,qp)
   !
#endif
   !
   call QP_Green_Function(qp,en,-1)
   !
   call QP_report_and_write(k,qp,en,-1)
   call local_free()
   return
   !
 endif
 !
 ! LIFETIMES
 !
 if (l_life) then
   !
   ! GW Lifetimes 
   !==============
   if (l_elel_corr)     call QP_real_axis(X,Xen,Xk,en,k,q,qp,Xw,Dip,0) 
   !
#if defined _QED
   !
   ! Radiative Lifetimes (Using <p>)
   !=================================
   if (l_elphoton_corr) call QED_Radiative_Lifetimes(en,k,q,qp,X,Dip)
   !
#endif
   !
   qp%Z=cONE
   !
   call QP_report_and_write(k,qp,en,-1)
   call local_free()
   return
   !
 endif
 !
 ! Reporting
 !
 call QP_report_and_write(k,qp,en,-1)
 call local_free()
 !
 contains
   !
   subroutine local_alloc()
     !
     YAMBO_ALLOC(qp%Z,(qp%n_states))
     YAMBO_ALLOC(qp%E,(qp%n_states))
     YAMBO_ALLOC(qp%E_bare,(qp%n_states))
     YAMBO_ALLOC(qp%k,(qp%nk,3))
     YAMBO_ALLOC(qp%table,(qp%n_states,3+n_sp_pol-1))
     !
     qp%table=QP_table
     qp%E=cZERO
     qp%k=0._SP
     !
     ! Sc energy steps. 2/3 If Newton/Secant. QP_Sc_steps 
     ! the full Green`s function is requested.
     !
     if (.not.l_life)                 QP_dSc_steps=2
     if (.not.l_life.and.QP_dSc_test) QP_dSc_steps=3
     !
     ! Cohsex is static
     !
     if (l_cohsex)                    QP_dSc_steps=1
     !
     if (trim(QP_solver)/='g')        QP_Sc_steps=QP_dSc_steps
     !
     YAMBO_ALLOC(QP_Sc,(QP_n_states,QP_Sc_steps))
     !
#if defined _ELPH
     if (l_elph_corr) then
       YAMBO_ALLOC(QP_OMS_Fan,(QP_n_states))
       YAMBO_ALLOC(QP_DW,(QP_n_states))
     endif
#endif
     !
     if (trim(QP_solver)=='g') then
       YAMBO_ALLOC(qp%GreenF,(QP_n_states,QP_Sc_steps))
       YAMBO_ALLOC(qp%S_total,(QP_n_states,QP_Sc_steps))
       YAMBO_ALLOC(qp%GreenF_W,(QP_n_states,QP_Sc_steps))
       qp%GreenF_n_steps=QP_Sc_steps
       YAMBO_ALLOC(QP_G_amplitude_integral,(QP_n_states))
       YAMBO_ALLOC(QP_G_info,(QP_n_states,3,2))
     else if (.not.l_life.and.QP_dSc_steps>1) then
       YAMBO_ALLOC(QP_dSc,(qp%n_states,QP_dSc_steps-1))
     endif
     !
   end subroutine
   !
   subroutine local_free()
     YAMBO_FREE(PAR_IND_WF_linear%element_2D)
     YAMBO_FREE(qp%Z)
     YAMBO_FREE(qp%E)
     YAMBO_FREE(qp%E_bare)
     YAMBO_FREE(qp%k)
     YAMBO_FREE(qp%table)
     YAMBO_FREE(QP_table)
     YAMBO_FREE(QP_state)
#if defined _ELPH
     YAMBO_FREE(QP_DW)
     YAMBO_FREE(QP_OMS_Fan)
#endif
     YAMBO_FREE(QP_Sc)
     YAMBO_FREE(QP_dSc)
     YAMBO_FREE(QP_Vnl_xc)
     YAMBO_FREE(QP_Vxc)
     YAMBO_FREE(qp%GreenF)
     YAMBO_FREE(qp%GreenF_W)
     YAMBO_FREE(qp%S_total)
     YAMBO_FREE(QP_G_info)
     YAMBO_FREE(QP_G_amplitude_integral)
     YAMBO_FREE(QP_G_zoom_er)
   end subroutine
   !
end subroutine
!
subroutine check_qp_states(QP_table,QP_state,qp,en,Xen)
  !
  ! Check if qp-indexes are continous for SC-GW
  !  
  use QP_m,          ONLY:QP_t
  use electrons,     ONLY:n_sp_pol,levels
#include<memory.h>
  !
  !
  type(QP_t), intent(in)      :: qp
  type(levels),intent(inout)  :: en,Xen
  logical,    intent(in)      :: QP_state(qp%nb,qp%nk)
  integer,    intent(in)      :: QP_table(qp%n_states,3+n_sp_pol-1)
  !
  ! Work space
  !
  integer :: lw_bnd
  !
  lw_bnd=minval(QP_table(:,1))
  !
  if(any(QP_state(lw_bnd:qp%nb,:).eqv..FALSE.).or.qp%nk/=en%nk) &
      & call error("Not continous QP-indexes not suppoerted in self-consistent GW")
  !
  if(.not.allocated(en%Eo)) then
    YAMBO_ALLOC(en%Eo,(en%nb,en%nk,n_sp_pol))
    en%Eo=en%E            
  endif
  !
end subroutine check_qp_states
