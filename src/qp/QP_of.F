!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): AM
!
subroutine QP_of(qp,en,QPdb_read_err,what)
 !
 use pars,          ONLY:lchlen
 use drivers,       ONLY:l_life,l_elph_corr
 use com,           ONLY:msg
 use electrons,     ONLY:levels,spin,n_sp_pol
 use QP_m,          ONLY:QP_t,QP_Vxc,QP_Sc,QP_Vnl_xc,QP_table,&
&                        QP_solver,QP_n_states,Vxc_kind,l_extended_output,&
&                        use_GreenF_to_eval_QP,Vnlxc_kind,report_Vxc,&
&                        report_Vnlxc,QP_G_amplitude_integral,&
&                        QP_states_simmetrize,QP_G_info,QP_dSc,report_Sc
 use OUTPUT,        ONLY:OUTPUT_driver
 use interfaces,    ONLY:DESC_write
 use stderr,        ONLY:intc
 !
 implicit none
 !
 type(QP_t)   ::qp
 type(levels) ::en
 integer      ::QPdb_read_err
 character(*) ::what
 !
 ! Work Space
 !
 integer           ::i_qp,i_w,qp_spin
 character(lchlen) ::G_Sc_name
 logical           ::state_is_2do(QP_n_states)
 !
 if ( trim(QP_solver)=='n'.or.trim(QP_solver)=='s'.or.l_life.or.&
&    (trim(QP_solver)=='g'.and.use_GreenF_to_eval_QP).or.&
&    trim(QP_solver)=='mixed'.or.what=='hf') then
   !
   call OUTPUT_driver(what,action="reset")
   call OUTPUT_driver(what,action="open")
   !
   call DESC_write('o '//what,'#',qp%desc)
   !
   if (report_Vxc)   call msg('o '//what,'#',' Vxc  ='//trim(Vxc_kind),INDENT=0)
   if (report_Vnlxc) call msg('o '//what,'#',' Vnlxc='//trim(Vnlxc_kind),INDENT=0)
   !
   if (index(what,"qp")>0) qp%k=0.
   !
   do i_qp=1,QP_n_states
     !
     ! Fill the COL type with values
     !===============================
     call OUTPUT_driver(what,TITLES=(/"K-point","Band   "/),I_VALUES=(/QP_table(i_qp,3),QP_table(i_qp,1)/))
     call OUTPUT_driver(what,TITLES=(/"Eo"/),R_VALUES=(/qp%E_bare(i_qp)/),UNIT="eV")
     if (allocated(qp%E)) then
       call OUTPUT_driver(what,TITLES=(/"E"/),R_VALUES=(/real(qp%E(i_qp))/),UNIT="eV")
       call OUTPUT_driver(what,TITLES=(/"E-Eo"/),R_VALUES=(/real(qp%E(i_qp)-qp%E_bare(i_qp))/),UNIT="eV")
       call OUTPUT_driver(what,TITLES=(/"E-Eo"/),R_VALUES=(/real(qp%E(i_qp)-qp%E_bare(i_qp))/),UNIT="meV")
       call OUTPUT_driver(what,TITLES=(/"E-Eo"/),R_VALUES=(/real(qp%E(i_qp)-qp%E_bare(i_qp))/),UNIT="MHz")
       call OUTPUT_driver(what,TITLES=(/"E-Eo"/),R_VALUES=(/real(qp%E(i_qp)-qp%E_bare(i_qp))/),UNIT="GHz")
       call OUTPUT_driver(what,TITLES=(/"Ehf"/),R_VALUES=(/real(qp%E(i_qp))/),UNIT="eV")
     endif
     if (allocated(QP_sc)) then
       call OUTPUT_driver(what,TITLES=(/"Sc|Eo"/),R_VALUES=(/real(QP_Sc(i_qp,1))/),UNIT="eV")
     endif
     if (allocated(QP_dsc)) then
       call OUTPUT_driver(what,TITLES=(/"Sc|E"/),&
&                             R_VALUES=(/real(QP_Sc(i_qp,1)+QP_dSc(i_qp,1)*(qp%E(i_qp)-qp%E_bare(i_qp)))/),UNIT="eV")
       call OUTPUT_driver(what,TITLES=(/"dSc/dw|Eo"/),R_VALUES=(/real(QP_dSc(i_qp,1))/),UNIT="eV")
       call OUTPUT_driver(what,TITLES=(/"dV/V[%]"/),R_VALUES=(/real(QP_dSc(i_qp,1))*100/))
     endif
     if (allocated(QP_Vxc))    call OUTPUT_driver(what,TITLES=(/"Vxc"/),R_VALUES=(/real(QP_Vxc(i_qp))/),UNIT="eV")
     if (allocated(QP_Vnl_xc)) call OUTPUT_driver(what,TITLES=(/"Vnlxc"/),R_VALUES=(/real(QP_Vnl_xc(i_qp))/),UNIT="eV")
     if (n_sp_pol==2) then
       if (spin(QP_table(i_qp,:))==1) qp_spin= 1
       if (spin(QP_table(i_qp,:))==2) qp_spin=-1
       call OUTPUT_driver(what,TITLES=(/"Spin_Pol"/),I_VALUES=(/qp_spin/))
     endif
     if (allocated(qp%Z)) then
       call OUTPUT_driver(what,TITLES=(/"Re(Z)","Im(Z)"/),R_VALUES=(/real(qp%Z(i_qp)),aimag(qp%Z(i_qp))/))
     endif
     call OUTPUT_driver(what,TITLES=(/"Width"/),R_VALUES=(/aimag(qp%E(i_qp))/),UNIT="meV")
     call OUTPUT_driver(what,TITLES=(/"Width"/),R_VALUES=(/aimag(qp%E(i_qp))/),UNIT="fs")
     call OUTPUT_driver(what,TITLES=(/"Width"/),R_VALUES=(/aimag(qp%E(i_qp))/),UNIT="ps")
     call OUTPUT_driver(what,TITLES=(/"Width"/),R_VALUES=(/aimag(qp%E(i_qp))/),UNIT="ns")
     if (allocated(QP_G_amplitude_integral)) then
       call OUTPUT_driver(what,TITLES=(/"Integral"/),R_VALUES=(/QP_G_amplitude_integral(i_qp)/))
     endif
     if (allocated(QP_G_info)) then
       call OUTPUT_driver(what,TITLES=(/"Integral"/),R_VALUES=(/QP_G_amplitude_integral(i_qp)/))
       call OUTPUT_driver(what,TITLES=(/"E_25%(low)"/),R_VALUES=(/QP_G_info(i_qp,1,1)/),UNIT="eV")
       call OUTPUT_driver(what,TITLES=(/"E_25%(high)"/),R_VALUES=(/QP_G_info(i_qp,1,2)/),UNIT="eV")
       call OUTPUT_driver(what,TITLES=(/"E_50%(low)"/),R_VALUES=(/QP_G_info(i_qp,2,1)/),UNIT="eV")
       call OUTPUT_driver(what,TITLES=(/"E_50%(high)"/),R_VALUES=(/QP_G_info(i_qp,2,2)/),UNIT="eV")
       call OUTPUT_driver(what,TITLES=(/"E_75%(low)"/),R_VALUES=(/QP_G_info(i_qp,3,1)/),UNIT="eV")
       call OUTPUT_driver(what,TITLES=(/"E_75%(high)"/),R_VALUES=(/QP_G_info(i_qp,3,2)/),UNIT="eV")
     endif
     !
     ! Select the colummns to print
     !==============================
     call OUTPUT_driver(what,ORDER=(/"K-poi","Band "/))
     !
     if (index(what,"qp")>0) then
       !
       if (l_life) then
         call OUTPUT_driver(what,ORDER=(/"Eo"/))
         call OUTPUT_driver(what,ORDER=(/"Width [me","Width [fs"/))
#if defined _QED
         call OUTPUT_driver(what,ORDER=(/"Width [ps","Width [ns"/))
#endif
       else
#if defined _QED
         call OUTPUT_driver(what,ORDER=(/"Eo"/))
         call OUTPUT_driver(what,ORDER=(/"E-Eo [meV","E-Eo [GHz"/))
         call OUTPUT_driver(what,ORDER=(/"Sc|Eo"/))
         call OUTPUT_driver(what,ORDER=(/"Re(Z)"/))
         call OUTPUT_driver(what,ORDER=(/"Width [me","Width [ps","Width [ns"/))
         call OUTPUT_driver(what,ORDER=(/"dV"/))
#else
         if (QPdb_read_err/=0) then
           if (trim(QP_solver)=='g'.and.use_GreenF_to_eval_QP) then
             call OUTPUT_driver(what,ORDER=(/"E [eV"/))
             call OUTPUT_driver(what,ORDER=(/"E-Eo [eV"/))
             call OUTPUT_driver(what,ORDER=(/"Width [me"/))
             call OUTPUT_driver(what,ORDER=(/"Re(Z)","Im(Z)"/))
             call OUTPUT_driver(what,ORDER=(/"E_25(l","E_25(h"/))
             call OUTPUT_driver(what,ORDER=(/"E_50(l","E_50(h"/))
             call OUTPUT_driver(what,ORDER=(/"E_75(l","E_75(h","Integr"/))
           else if (l_extended_output) then
             call OUTPUT_driver(what,ORDER=(/"Eo"/))
             call OUTPUT_driver(what,ORDER=(/"E [eV"/))
             call OUTPUT_driver(what,ORDER=(/"E-Eo [eV"/))
             call OUTPUT_driver(what,ORDER=(/"Vxc  ","Vnlxc"/))
             call OUTPUT_driver(what,ORDER=(/"Sc|Eo"/))
             call OUTPUT_driver(what,ORDER=(/"Sc|E [e"/))
             call OUTPUT_driver(what,ORDER=(/"dSc/dw"/))
             call OUTPUT_driver(what,ORDER=(/"Re(Z)","Im(Z)"/))
             call OUTPUT_driver(what,ORDER=(/"Width [me","Width [fs"/))
           else 
             call OUTPUT_driver(what,ORDER=(/"Eo"/))
             if (.not.report_Sc) call OUTPUT_driver(what,ORDER=(/"E [eV"/))
             call OUTPUT_driver(what,ORDER=(/"E-Eo [eV"/))
             if (     report_Sc) call OUTPUT_driver(what,ORDER=(/"Sc|Eo"/))
             if (.not.report_Sc) call OUTPUT_driver(what,ORDER=(/"Re(Z)"/))
             if (l_elph_corr) then
               call OUTPUT_driver(what,ORDER=(/"Width [me"/))
             endif
           endif
         else
           call OUTPUT_driver(what,ORDER=(/"Eo"/))
           call OUTPUT_driver(what,ORDER=(/"E [eV"/))
           call OUTPUT_driver(what,ORDER=(/"E-Eo [eV"/))
           call OUTPUT_driver(what,ORDER=(/"Vxc  ","Vnlxc"/))
           call OUTPUT_driver(what,ORDER=(/"Re(Z)","Im(Z)"/))
           call OUTPUT_driver(what,ORDER=(/"Width [me","Width [fs"/))
         endif
#endif
       endif
       !
     endif
     !
     ! HF
     !----
     if (what=="hf") then
       call OUTPUT_driver(what,ORDER=(/"Eo ","Ehf"/))
       if (report_Vxc)   call OUTPUT_driver(what,ORDER=(/"Vxc"/))
       if (report_Vnlxc) call OUTPUT_driver(what,ORDER=(/"Vnlxc"/))
     endif
     !
     if (n_sp_pol==2) call OUTPUT_driver(what,ORDER=(/"Spin"/))
     !
     call OUTPUT_driver(what,action="write")
     !
   enddo
   !
   call OUTPUT_driver(what,action="close")
   !
 else if (trim(QP_solver)=='g') then
   !
   call QP_states_simmetrize(en,state_is_2do=state_is_2do)
   !
   do  i_qp=1,QP_n_states
     !
     G_Sc_name='G_Sc_band_'//trim(intc(QP_table(i_qp,1)))//'_k_'//trim(intc(QP_table(i_qp,3)))
     if (n_sp_pol>1) G_Sc_name=trim(G_Sc_name)//'_s'//trim(intc(spin(QP_table(i_qp,:))))
     !
     call OUTPUT_driver(trim(G_sc_name),action="reset")
     call OUTPUT_driver(trim(G_sc_name),action="open")
     !
     call msg('o G_Sc','# GW [Green`s function & Self-Energy]')
     call DESC_write('o '//trim(G_sc_name),'#',qp%desc)
     !
     if (allocated(QP_G_amplitude_integral)) then
       call msg('o G_Sc','#')
       call msg('o G_Sc','# Spectral function at this point integrates to',QP_G_amplitude_integral(i_qp),INDENT=0)
       call msg('o G_Sc','#')
     endif
     !
     do i_w=1,qp%GreenF_n_steps
       call OUTPUT_driver(trim(G_sc_name),TITLES=(/"Energy","IM(En)"/),&
                              R_VALUES=(/real(qp%GreenF_W(i_qp,i_w)),aimag(qp%GreenF_W(i_qp,i_w))/),UNIT="eV")
       call OUTPUT_driver(trim(G_sc_name),TITLES=(/"Re(G)","Im(G)"/),&
&                         R_VALUES=(/real(qp%GreenF(i_qp,i_w)),aimag(qp%GreenF(i_qp,i_w))/),UNIT="eVm1")
       if (allocated(QP_Vxc).and.allocated(QP_Vnl_xc)) then
         call OUTPUT_driver(trim(G_sc_name),TITLES=(/"Re(S_c)"/),&
&                           R_VALUES=(/real(qp%S_total(i_qp,i_w)-QP_Vnl_xc(i_qp)+QP_Vxc(i_qp))/),UNIT="eV")
       endif
       call OUTPUT_driver(trim(G_sc_name),TITLES=(/"Re(S_tot)","Im(S_c)  "/),&
&                         R_VALUES=(/real(qp%S_total(i_qp,i_w)),aimag(qp%S_total(i_qp,i_w))/),UNIT="eV")
       !
       call OUTPUT_driver(trim(G_sc_name),ORDER=(/"Energy","Im(En)"/))
       call OUTPUT_driver(trim(G_sc_name),ORDER=(/"Re(G)","Im(G)"/))
       call OUTPUT_driver(trim(G_sc_name),ORDER=(/"Re(S_tot)","Im(S_c)  "/))
       if (report_Vnlxc) call OUTPUT_driver(trim(G_sc_name),ORDER=(/"Re(S_c)"/))
       !
       call OUTPUT_driver(trim(G_sc_name),action="write")
       !
     enddo
     !
     call OUTPUT_driver(trim(G_sc_name),action="close")
     !
   enddo
   !
 endif
 !
end subroutine
