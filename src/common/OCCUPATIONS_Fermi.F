!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): AM
!
subroutine OCCUPATIONS_Fermi(E,K,OBJ,mode,n_bands,impose_balance)
 !
 ! Mode = "FERMI" 
 !  Update the Fermi energy
 !  Output : E%E_fermi
 !
 ! Mode =  "OCCUPATIONS" (implies "FERMI")
 !  Define E%f and reports variations in Ef/nbf/nbm
 !  Output : E%nbf,E%nbm
 !  E%E are shifted 
 !
 ! no Mode  (implies "FERMI" and "OCCUPATIONS")
 ! Full report of system electronic character
 !
 use pars,          ONLY:SP,DP
 use units,         ONLY:HA2KEL
 use drivers,       ONLY:Finite_Tel
 use D_lattice,     ONLY:Tel,T_elecs,T_holes,Bose_Temp
 use electrons,     ONLY:levels,nel,n_sp_pol,spin_occ,n_spinor,n_spin
 use R_lattice,     ONLY:bz_samp
 use vec_operate,   ONLY:sort
 use functions,     ONLY:Fermi_fnc
 use interfaces,    ONLY:OCCUPATIONS_Gaps
#if defined _SC
 use SC,            ONLY:SC_neq_kind,SC_mu,SC_occ
#endif 
 !
#include<memory.h>
 !
 type(levels)  ::E
 type(bz_samp) ::K
 character(*)           ::OBJ
 character(*), OPTIONAL ::mode
 integer,      OPTIONAL ::n_bands
 logical,      OPTIONAL ::impose_balance
 !
 ! Work Space
 !
 integer :: i1,is,ib,ik,i_trials(2),n_b_full_guess,i_Ef(2),i_Ef_fine(2),n_total_states,group_size,nb_here
 integer :: index(E%nk*E%nb*n_sp_pol,3),index_E_sorted(E%nk*E%nb*n_sp_pol)
 real(SP):: E_sorted(E%nk*E%nb*n_sp_pol),n_of_el_from_zero(2),Ef(2),nel_DbGd,nel_diff_zero,n_elecs,n_holes,balance
 logical :: Fermi_is_converged,impose_balance_
 real(SP),parameter :: Tel_step=0.1_SP/HA2KEL
 real(SP) :: n_carr(2),mu_zero,mu_hole,mu_el
 !
 !     n_spin n_sp_pol n_spinor  spin_occ 
 !       1        1        1         2      
 !       2        1        2         1      non collinear
 !       2        2        1         1          collinear
 !
 spin_occ=2.0_SP/real(n_spin)
 !
 ! All energies are stored in E_sorted...
 !
 nb_here=E%nb
 if (present(n_bands)) nb_here=n_bands
 !
 i1=0
 do ik=1,E%nk
   do ib=1,nb_here
     do is=1,n_sp_pol
       i1=i1+1
       index(i1,:)=(/ib,ik,is/)
       E_sorted(i1)=E%E(ib,ik,is)
     enddo
   enddo
 enddo
 n_total_states=E%nk*nb_here*n_sp_pol
 !
 ! ... and sorted
 !
 call sort(arrin=E_sorted(:n_total_states),indx=index_E_sorted(:n_total_states))
 !
 ! First guess
 !
 ! real(nel)/2.0_SP*real(n_spinor) is last occupied state
 !
 ! n_b_full_guess is used to avoid spurious oscillations
 ! in the Fermi Level search for system with a gap.
 ! In general those oscillations appear when the gap is small.
 !
 n_b_full_guess= nint( real(nel)/2.0_SP*real(n_spinor)+.1)
 !
 if (n_b_full_guess+1 > nb_here ) call error(' Not enough states to accomodate all electrons. Include more states in the DFT run.')
 !
 Ef(1)=maxval(E%E(n_b_full_guess,:,:))
 Ef(2)=minval(E%E(n_b_full_guess+1,:,:))
 !
 ! n_b_full_guess not set when the system is metallic ...
 ! 
 if (Ef(2)<Ef(1).or.Ef(1)==Ef(2)) then
   Ef(1)=minval(E%E(n_b_full_guess,:,:))
   Ef(2)=maxval(E%E(n_b_full_guess,:,:))
   n_b_full_guess=0
 endif
 !
 ! ... or when there is a odd number of electrons (when n_sp_pol>1
 ! the system can be semiconductive in the two channels).
 ! 
 if (mod(nel,2._SP)/=0._SP) n_b_full_guess=0
 !
 ! Start Loop
 ! 
 i_trials=1
 nel_diff_zero=1.E-5
 Fermi_is_converged=.false.
 !
 ! BUG-fix [Andrea 1/6/2012]: when the system is semiconductive it may happen the
 ! Efermi falls below or above the CBM/VBM of the Double Grid energy levels. To avoid this
 ! a very small temperature is imposed forcing the Fermi level to fall within the
 ! gap.
 !
 if (E%FG%nb>0.and.Tel<Tel_step) Tel=Tel_step
 !
 FERMI_LOOP: do while (.not.Fermi_is_converged)
   !
   i_Ef=(/1,n_total_states/)
   i_trials(1)=i_trials(1)+1 
   !
   ! 1. [Rough search] by looping of group_size elements
   !
   group_size=max(1,int(n_total_states/500))
   do i1=1,n_total_states,group_size
     call ef2nel(Tel,E_sorted(i1),n_of_el_from_zero(1),'both ')
     if (n_of_el_from_zero(1)<-nel_diff_zero) i_Ef(1)=i1
     if (n_of_el_from_zero(1)> nel_diff_zero) then
       i_Ef(2)=i1
       exit
     endif
   enddo
   !
   i_Ef_fine=i_Ef
   !
   if (group_size>1) then
     !
     ! 2. [Finer search] by looping of 1 element
     !
     do i1=i_Ef(1),i_Ef(2)
       call ef2nel(Tel,E_sorted(i1),n_of_el_from_zero(1),'both ')
       if (n_of_el_from_zero(1)<-nel_diff_zero) i_Ef_fine(1)=i1
       if (n_of_el_from_zero(1)> nel_diff_zero) then
         i_Ef_fine(2)=i1
         exit
       endif
     enddo
     !
   endif
   !
   if (i_Ef_fine(2)==n_total_states) call error('Not enough states to converge the Fermi Level')
   !
   ! 3. Is any level in the Ef_fine range ok?
   !
   do i1=i_Ef_fine(1),i_Ef_fine(2)
     call ef2nel(Tel,E_sorted(i1),n_of_el_from_zero(1),'both ')
     if (abs(n_of_el_from_zero(1))<nel_diff_zero.and.E_sorted(i1+1)-E_sorted(i1)>Tel_step) then
       E%E_Fermi=E_sorted(i1)
       ! To Code!
       !E%E_Fermi=E_sorted(i1)+E_sorted(i1+1))/2.
       exit FERMI_LOOP
     endif
   enddo
   !
   ! 4. extend i_Ef_fine to include degenerate states
   !
   i_Ef=i_Ef_fine
   do i1=i_Ef_fine(1)-1,1,-1
     if (E_sorted(i_Ef_fine(1))-E_sorted(i1)<Tel_step) i_Ef(1)=i1
     if (E_sorted(i_Ef_fine(1))-E_sorted(i1)>Tel_step) exit
   enddo
   do i1=i_Ef_fine(2)+1,n_total_states
     if (E_sorted(i1)-E_sorted(i_Ef_fine(2))<Tel_step) i_Ef(2)=i1
     if (E_sorted(i1)-E_sorted(i_Ef_fine(2))>Tel_step) exit
   enddo
   if (i_Ef(1)/=i_Ef(2).or.i_Ef(1)/=i_Ef_fine(2)) then
     Tel=Tel+Tel_step
     i_Ef(1)=max(1,i_Ef(1)-2)
     i_Ef(2)=min(i_Ef(2)+2,n_total_states)
   endif
   !
   ! 5. Use a secant-like procedure to converge to the final Level
   !
   Ef=(/E_sorted(i_Ef(1)),E_sorted(i_Ef(2))/)
   !
   call ef2nel(Tel,Ef(1),n_of_el_from_zero(1),'both ')
   call ef2nel(Tel,Ef(2),n_of_el_from_zero(2),'both ')
   !
   E%E_Fermi=Ef(1)
   if (n_of_el_from_zero(2)<-n_of_el_from_zero(1)) E%E_Fermi=Ef(2)
   !
   i_trials(2)=1
   do while (i_trials(2)<=1000)
     call ef2nel(Tel,E%E_Fermi,n_of_el_from_zero(1),'both ')
     if (n_of_el_from_zero(1)<-nel_diff_zero) Ef(1)=E%E_Fermi
     if (n_of_el_from_zero(1)> nel_diff_zero) Ef(2)=E%E_Fermi
     if (abs(n_of_el_from_zero(1))<=nel_diff_zero) then
       Fermi_is_converged=.true.
       exit FERMI_LOOP
     endif
     if (abs(Ef(1)-Ef(2))<1.E-8) exit FERMI_LOOP
     E%E_Fermi=(Ef(1)+Ef(2))/2.
     i_trials(2)=i_trials(2)+1
   enddo
   !
   if (Fermi_is_converged) exit FERMI_LOOP
   !
   Tel=Tel+Tel_step
   if (i_trials(1)>100) nel_diff_zero=nel_diff_zero*10.
   if (i_trials(1)==1000) call error('Impossible to converge the Fermi Level')
   !
 enddo FERMI_LOOP
 !
 i_trials(1)=1
 T_elecs=Tel
 T_holes=Tel
 !
 impose_balance_=.false.
 if(present(impose_balance)) impose_balance_=impose_balance
 if (impose_balance_) then
   call ef2nel(T_holes,E%E_Fermi,n_holes,'holes')
   do while (i_trials(1)<5000)
     call ef2nel(T_elecs,E%E_Fermi,n_elecs,'elecs')
     if(abs(n_elecs+n_holes)==0._SP) exit
     balance=(n_holes-n_elecs)/(n_elecs+n_holes)
     if ( abs(balance)< 1.E-5 ) exit
     if (     balance <-1.E-5 ) T_elecs=T_elecs-Tel_step*abs(n_holes-n_elecs)*1000
     if (     balance > 1.E-5 ) T_elecs=T_elecs+Tel_step*abs(n_holes-n_elecs)*1000
     i_trials(1)=i_trials(1)+1
     if (i_trials(1)==5000) then
       T_elecs=Tel
       call warning(' Impossible to balance electrons and holes number')
     endif
   enddo
 endif
 !
 !
 ! Mode = 0. Check only for the Fermi energy
 !
 if (Bose_Temp<0.) Bose_Temp=Tel
 Finite_Tel=any((/Tel,Bose_Temp/)>=Tel_step*3._SP)
 !
 if (present(mode)) then
   if (mode=="FERMI") return
 endif
 !
 ! Build the occupations
 !
 if (.not.allocated(E%f)) then  
   YAMBO_ALLOC(E%f,(nb_here,E%nk,n_sp_pol))
 endif
 !
 mu_zero=E%E_fermi
 mu_el  =E%E_fermi
 mu_hole=E%E_fermi
#if defined _SC
 n_carr=0.
 if (trim(SC_neq_kind)=="constrained-mu") then
   write(*,*) "Ef= ",E%E_fermi*27.2116
   mu_zero= SC_mu(1)
   mu_hole= SC_mu(2)
   mu_el  = SC_mu(3)
 endif
#endif
 !
 do i1=1,n_total_states
   ib=index( index_E_sorted(i1) , 1)
   ik=index( index_E_sorted(i1) , 2)
   is=index( index_E_sorted(i1) , 3)
   !
   if(E_sorted(i1)<=mu_zero) E%f(ib,ik,is)=spin_occ*Fermi_fnc(E_sorted(i1)-mu_hole,T_holes)
   if(E_sorted(i1)> mu_zero) E%f(ib,ik,is)=spin_occ*Fermi_fnc(E_sorted(i1)-mu_el,  T_elecs)
   !
#if defined _SC
   if((E%E(ib,ik,is)-mu_zero)<=0._SP) n_carr(1)=n_carr(1)+(spin_occ-E%f(ib,ik,is))*K%weights(ik)
   if((E%E(ib,ik,is)-mu_zero)> 0._SP) n_carr(2)=n_carr(2)+          E%f(ib,ik,is) *K%weights(ik)
#endif
   !
 enddo
 !
#if defined _SC
 !
 if (trim(SC_neq_kind)=="constrained-occ") then
   do is=1,n_sp_pol
     do ik=1,E%nk
       do ib=1,nb_here
         if( SC_occ(ib,ik,is)==-99._SP ) cycle
         E%f(ib,ik,is)=SC_occ(ib,ik,is)*spin_occ
       enddo
     enddo
   enddo
 endif
 !
#endif
 !
 E%E(:,:,:)=E%E(:,:,:)-E%E_Fermi
 !
 ! => Double Grid extension
 !
 call OCCUPATIONS_Extend_Double_Grid(OBJ,E,K)
 !
 ! Gaps, VBM/CBM
 !
 if (present(mode)) then
   if (mode=="OCCUPATIONS") then
     call OCCUPATIONS_Gaps(E)
     return
   endif
 endif
 !
 call REPORT_Occupations(OBJ,E)
 !
 contains
   !
   subroutine ef2nel(tTel,Ef,N_of_el_diff,what)
     !
     implicit none
     !
     real(SP),     intent(in)  :: tTel,Ef
     real(SP),     intent(out) :: N_of_el_diff
     character(5), intent(in)  :: what
     !
     !Work Space
     !
     integer  :: i1,ik,ib 
     real(DP) :: nel_acc,nel_theo,elecs_acc,holes_acc,tmp_holes,tmp_elecs
     !
     nel_acc=0._DP
     nel_theo=0._DP
     holes_acc=0._DP
     elecs_acc=0._DP
     do i1=1,n_total_states
       ib=index( index_E_sorted(i1) , 1)
       ik=index( index_E_sorted(i1) , 2)
       is=index( index_E_sorted(i1) , 3)
       if (ib<=n_b_full_guess) nel_theo=nel_theo+real(spin_occ*K%weights(ik),DP)
       tmp_holes= real(spin_occ*Fermi_fnc(-(E_sorted(i1)-Ef),tTel)*K%weights(ik),DP)
       tmp_elecs= real(spin_occ*Fermi_fnc( (E_sorted(i1)-Ef),tTel)*K%weights(ik),DP)
       if(what=='both ') nel_acc=nel_acc+tmp_elecs
       if(what=='holes'.and.E_sorted(i1)<Ef) holes_acc=holes_acc+tmp_holes
       if(what=='elecs'.and.E_sorted(i1)>Ef) elecs_acc=elecs_acc+tmp_elecs
     enddo
     if (n_b_full_guess==0) nel_theo=nel
     if(what=='both ') N_of_el_diff=real(nel_acc-nel_theo,SP)
     if(what=='holes') N_of_el_diff=real(holes_acc,SP)
     if(what=='elecs') N_of_el_diff=real(elecs_acc,SP)
     !
   end subroutine
   !
end subroutine
