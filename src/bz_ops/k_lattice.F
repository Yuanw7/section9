!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): AM
!
integer function k_lattice(k,Xk,G_shell,be_verbose)
 !
 ! Output(s)
 !
 !  k_grid_b k_grid_dim k_grid k_grid_uc_vol
 !
 use pars,           ONLY:SP,schlen,zero_dfl
 use vec_operate,    ONLY:v_norm,v_is_zero,cross_product,&
&                         c2a,sort,rlu_v_norm
 use com,            ONLY:msg
 use matrix_operate, ONLY:m3det,m3inv
 use R_lattice,      ONLY:b,k_grid_b,k_grid_dim,k_grid,RL_vol,&
&                         k_grid_uc_vol,nkibz,nkbz,nXkibz,nXkbz,d3k_factor,&
&                         bz_samp,g_vec,ng_in_shell,Xk_grid_new_mapping
 use zeros,          ONLY:k_rlu_zero
 use parser_m,       ONLY:parser
 use drivers,        ONLY:l_rim_w
 !
#include<memory.h>
 !
 type(bz_samp)::k,Xk
 integer      ::G_shell
 logical      ::be_verbose
 !
 ! Work Space
 !
 integer  :: i1,i2,i3,iv(3),ig,N_g_vecs,N_ext_K_pts,k_grid_borders(3,2)
 real(SP) :: v0(3),v1(3),v0_ibz_mod,k_b_m1(3,3),min_idist(2)
 real(SP),allocatable ::idist(:),extended_k_points(:,:)
 integer, allocatable ::order(:)
 character(schlen)    ::warn_msg,ch 
 logical  :: small_b_in_UC
 !
 ! ################## NEW MAPPING ######################
 real(SP) :: q_step(3),k_shift(3)
 integer  :: id
 ! ################## NEW MAPPING ######################
 !
 k_lattice=0
 !
 ! First I define the Xk type from the first nXkibz kpts of k
 !
 k%description ='k'
 Xk%description='Xk'
 Xk%nibz       =nXkibz
 !
 if (G_shell == 1) then
   !
   YAMBO_ALLOC(Xk%pt,(Xk%nibz,3))
   Xk%pt(:,:)=k%pt(:Xk%nibz,:)
   !
   ! IBZ -> BZ
   !
   call k_expand(k)
   call k_expand(Xk)
   !
 endif
 !
 call k_ibz2bz(Xk,'a',.true.)
 !
 nkibz     =k%nibz
 nkbz      =k%nbz
 nXkbz     =Xk%nbz
 d3k_factor=RL_vol/real(Xk%nbz)
 !
 N_g_vecs=ng_in_shell(G_shell)
 !
 ! Andrea 31/03/2009. Bug pointed out by a user.
 !
 ! writing this routine I assumed that a subgroup of the
 ! k-points constituets a uniform grid .
 ! Of course this is not true if you pick up gamma plus
 ! a point exactly on the BZ border. As a first work around
 ! I extended the k-grid using a certain number of 
 ! RL vectors shells. (passed via G_shell)
 !
 YAMBO_ALLOC(extended_k_points,(nXkbz*N_g_vecs,3))
 N_ext_K_pts=0
 do ig=1,N_g_vecs
   call c2a(b,v_in=g_vec(ig,:),v_out=v1,mode='ki2a')
   do i1=1,Xk%nbz
     N_ext_K_pts=N_ext_K_pts+1
     extended_k_points(N_ext_K_pts,:)= Xk%ptbz(i1,:)-Xk%ptbz(1,:)+ v1
   enddo
 enddo
 !
 YAMBO_ALLOC(idist,(N_ext_K_pts))
 YAMBO_ALLOC(order,(N_ext_K_pts))
 !
 min_idist=1.E10
 !
 do i1=1,N_ext_K_pts
   !
   v0=extended_k_points(i1,:)
   !
   ! Andrea (5/3/2010): Here I am searching for the smallest
   ! non linear-dependent vectors of the K grid (shifted in gamma).
   ! Up to today "smallest" meant in chartesian coordinates. However I want
   ! to build-up a grid in RLU and in the unlucky case where the smallest in 
   ! chartesian is not the smallest in RLU the grid will turn out to be not correctly sized.
   !
   ! At the moment I cannot judge if this change will or not yield other problems in the case.
   ! In any case the deep checks in bz_samp_indexes should prevent them.
   !
   ! Andrea (29/12/2011). The 2D surface of the tutorials still does not work. And this is due to 
   ! the change introduced in 2010. Therefore I try to use the method that yields the smallest
   ! norm.
   !
   ! In the case of RIM-W, I need a grid which is the smallest in the rlu case
   !
   if(.not.l_rim_w) then
     do i2=1,2
       !
       if (i2==1) then
         if (v_is_zero(v0,zero_=k_rlu_zero)) cycle
         min_idist(1)=min(min_idist(1), rlu_v_norm(v0,inside_bz=.true.))
         min_idist(2)=min(min_idist(2), v_norm(v0))
       else if (min_idist(1)<min_idist(2)) then
         idist(i1)=rlu_v_norm(v0,inside_bz=.true.)
       else if (min_idist(1)>=min_idist(2)) then
         idist(i1)=v_norm(v0)
       endif
       order(i1)=i1
     enddo
   else
     ! Store only rlu distance
     idist(i1)=v_norm(v0)
     order(i1)=i1
     !
   endif
 enddo
 if (N_ext_K_pts>1) call sort(idist,indx=order)
 !
 YAMBO_FREE(idist)
 !
 k_grid_dim=0
 k_grid_b=0._SP
 do i1=1,N_ext_K_pts
   v0=extended_k_points(order(i1),:)
   !
   v0_ibz_mod=v_norm(v0) 
   !
   if (v0_ibz_mod<=1.E-5) cycle
   select case(k_grid_dim)
     case(0)
       k_grid_dim=1
       k_grid_b(1,:)=v0
       cycle
     case(1)
       v1=v0-proj(v0,k_grid_b(1,:))*k_grid_b(1,:)
       if (v_is_zero(v1,zero_=k_rlu_zero)) cycle
       k_grid_dim=2
       k_grid_b(2,:)=v0
       cycle
     case(2)
       k_grid_b(3,:)=v0
       !
       ! Here I check the vectors are independent.
       ! I use k_grid_b/v_module(k_grid_b(1,:)) to avoid small numbers
       ! when a large set of k-points is used
       !
       if (abs(m3det(k_grid_b))/v_norm(k_grid_b(1,:))<1.E-5)then
         k_grid_b(3,:)=0._SP
         cycle
       endif
       k_grid_dim=3
       cycle
     case(3)
       exit
    end select
 enddo
 YAMBO_FREE(order)
 YAMBO_FREE(extended_k_points)
 k_grid=0
 k_grid(k_grid_dim+1:)=1
 !
 if (k_grid_dim/=3) then
   !
   ! If K-grid dim/=3 I need to define the dim-1 k_grid_b to be
   ! the corresponding RL b (For example in a 2D grid in the
   ! x,y plane k_grid_b(3,:)\proto(0,0,1)
   !
   i3=0
   do i1=1,3
     v1(1)=0._SP
     do i2=1,k_grid_dim
       v1(1)=v1(1)+abs(proj(b(i1,:),k_grid_b(i2,:)))
     enddo
     if (abs(v1(1))<=1.E-5) then
       i3=i3+1
       if (i3+k_grid_dim>3) then
         warn_msg='Impossible to define the grid unit vectors'
         goto 1
       endif
       !
       ! Note that k_grid_b here is rlu (as output is cc) because I am using 
       ! the K components to build it.
       !
       call c2a(b,b(i1,:),k_grid_b(k_grid_dim+i3,:),'kc2a')
     endif
   enddo
   if (i3==0) then
     warn_msg='Impossible to define the grid unit vectors'
     goto 1
   endif
 endif
 !
 if (k_grid_dim==0.and.Xk%nbz/=1) then
   warn_msg='The grid is 0D but I have more than 1 kpt.'
   goto 1
 endif 
 !
 ! Check Net size to store all BZ KPTS in rlu coordinates
 ! 
 ! Andrea 31/03/2009. 
 !
 ! I have rewritten more clearly these lines where the
 ! size of the grid is estabilshed.
 !
 k_b_m1(:,1)=k_grid_b(1,:)
 k_b_m1(:,2)=k_grid_b(2,:)
 k_b_m1(:,3)=k_grid_b(3,:)
 k_grid_borders(:,1)= 100
 k_grid_borders(:,2)=-100
 if (abs(m3det(k_b_m1))>1.E-7) then
   call m3inv(M=k_b_m1)
   do i1=1,Xk%nbz
     v1=matmul(k_b_m1,Xk%ptbz(i1,:)-Xk%ptbz(1,:)) 
     iv=nint(v1)
     !
     ! The borders are the min/max value of the coefficients of
     ! the k-pts in terms of the grid unit vectors
     !
     do i2=1,3
       k_grid_borders(i2,1)=min( k_grid_borders(i2,1), iv(i2) )
       k_grid_borders(i2,2)=max( k_grid_borders(i2,2), iv(i2) )
     enddo
   enddo
 else
   warn_msg='Error in determing the shape of the K-grid'
   goto 1
 endif
 do i1=1,k_grid_dim
   !
   ! When the min and max border is zero it means the 
   ! the corresponding unit cell vector is a RL vector (G_shells>1)
   !
   if (all(k_grid_borders(i1,:)/=0)) then
     !
     ! Exclude the max index when both borders are on the border of the
     ! rlu unit cell
     ! 
     small_b_in_UC=.FALSE.
     !
     do while (.not.small_b_in_UC)
       v0=(k_grid_borders(i1,2)-k_grid_borders(i1,1))*abs(k_grid_b(i1,:))
       if (any(v0>1.-1.E-5)) then
         k_grid_borders(i1,2)=k_grid_borders(i1,2)-1
       else
         small_b_in_UC=.TRUE.
       endif
     enddo
     !
   endif
   !
   if (k_grid(i1)==0) k_grid(i1)=1
   !
   ! Size of the shifted grid (used by bz_samp_indexes)
   ! 
   k_grid(i1)=k_grid_borders(i1,2)-k_grid_borders(i1,1)+1
   !
 enddo
 !
 ! ################ BEGIN NEW MAPPING ##################
 !
 if(Xk_grid_new_mapping) then
   !
   call k_find_smallest_q(Xk%ptbz,q_step,Xk%nbz)
   !
   k_shift=Xk%ptbz(1,:) ! I assume the first k-point as shift
   !
   Xk%ptbz(:,:)=Xk%ptbz(:,:)-nint(Xk%ptbz(:,:)) ! Bring in the BZ
   !  
   ! Bring k-points between [0,1)
   !
   where(Xk%ptbz(:,:)<-zero_dfl)
     Xk%ptbz(:,:)=1._SP+Xk%ptbz(:,:)     
   end where
   !
   do id=1,3
     if(q_step(id)==0._SP) then
       k_grid(id)=1
     else
        k_grid(id)=nint(maxval(Xk%ptbz(:,id)-k_shift(id))/q_step(id))+1
     endif
   enddo
   !
 endif
 !
 ! ################  END  NEW MAPPING ##################
 !
 ! Reporting
 !
 write (ch,'(i1,a)') k_grid_dim,'D'
 call msg('r','Compatible Grid is   ',trim(ch))
 !
 if (k_grid_dim>0) then
   call msg('r','Base K vectors       ',(/' K_min[ 1 ]  ',' K_min[ 2 ]  ',' K_min[ 3 ]  '/))
   call msg('r','K_min[ 1 ] ',k_grid_b(1,:),"[rlu]")
 endif
 if (k_grid_dim>1) call msg('r','K_min[ 2 ] ',k_grid_b(2,:),"[rlu]")
 if (k_grid_dim>2) call msg('r','K_min[ 3 ] ',k_grid_b(3,:),"[rlu]")
 !
 ! Note that k_grid_b are Cartesian.
 !
 call c2a(b_in=b,v_in=k_grid_b(1,:),mode='ka2c')
 call c2a(b_in=b,v_in=k_grid_b(2,:),mode='ka2c')
 call c2a(b_in=b,v_in=k_grid_b(3,:),mode='ka2c')
 k_grid_uc_vol=0._SP
 v0=cross_product(k_grid_b(2,:),k_grid_b(3,:))
 do i1=1,3
   k_grid_uc_vol=k_grid_uc_vol+k_grid_b(1,i1)*v0(i1)
 enddo
 k_grid_uc_vol=abs(k_grid_uc_vol)
 !
 if (k_grid_dim>0) call msg('rs','Grid dimensions      ',k_grid(:k_grid_dim))
 call msg('r',                   'K lattice UC volume  ',k_grid_uc_vol,"[a.u.]")
 !
 ! CLEAN
 !
 call k_ibz2bz(Xk,"d",.false.)
 call k_ibz2bz(Xk,'i',.true.)
 return
 !
1 continue
 !
 if (len_trim(warn_msg)>0) then
   if (be_verbose) call warning(warn_msg) 
   k_lattice=-1
 endif
 call k_ibz2bz(Xk,"d",.false.)
 !
 call k_ibz2bz(Xk,'i',.true.)
 !
 contains
   !
   function proj(v1,v2)
     implicit none
     real(SP) :: v1(3),v2(3),proj
     proj=0.
     if (v_is_zero(v2,zero_=k_rlu_zero)) return
     proj=dot_product(v1,v2)/v_norm(v2)**2
   end function
   !
end function
