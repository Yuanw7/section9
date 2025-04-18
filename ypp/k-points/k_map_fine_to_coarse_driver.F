!
! License-Identifier: GPL
!
! Copyright (C) 2016 The Yambo Team
!
! Authors (see AUTHORS file for details): AM DS
!
subroutine k_map_fine_to_coarse_driver(Xk,Xen)
 !
 use pars,                ONLY:SP
 use stderr,              ONLY:intc,STRING_same
 use R_lattice,           ONLY:bz_samp,nkibz,bz_samp_reset,BZ_FineGd_mode,bz_samp_FineGd_reset
 use com,                 ONLY:msg
 use YPPm,                ONLY:n_FineGd_DBs
 use electrons,           ONLY:levels,FineGd_E_components_reset
 use IO_int,              ONLY:io_control
 use IO_m,                ONLY:REP,OP_WR_CL
 !
#include<memory.h>
 !
 type(bz_samp) :: Xk
 type(levels)  :: Xen
 !
 ! Work Space
 !
 type(bz_samp)        :: FineGd_k
 integer              :: i_err,ID,i_fg
 integer, external    :: io_Double_Grid
 logical              :: l_mix,l_expand,l_unexpand,FineGd_is_expanded
 !
 if (n_FineGd_DBs==0) then
   call msg("s",'DbGd: no database secified. Nothing to do.')
   return
 endif
 !
 if (Xk%FGbz%N>0) then
   call msg("s",'DbGd: database components already allocated. Nothing to do.')
   return
 endif
 !
 call section('*','Double Grid DB(s)')
 !
 ! Initial checks
 !================
 call section('+','Double Grid initialization')
 call k_map_fine_to_coarse_init(Xk,Xen,FineGd_k,FineGd_is_expanded,.FALSE.)
 !
 l_mix     =STRING_same(BZ_FineGd_mode,"mixed")
 l_expand  =STRING_same(BZ_FineGd_mode,"expanded")
 l_unexpand=STRING_same(BZ_FineGd_mode,"unexpanded")
 !
 ! Expand the Coarse Grid 
 !========================
 call k_build_up_BZ_tables(Xk)
 call k_ibz2bz(Xk,'i',.true.)
 !
 ! FineGd E to IBZ map
 !=======================
 call section('=','Double Grid Energies-IBZ mapping and Fine Grid re-ordering')
 call k_map_fine_to_coarse('Energies-IBZ',Xk,Xk%FGbare,FineGd_k,Xen)
 !
 ! Expand the Fine Grid 
 !========================
 !
 if (l_mix.or.l_expand) then
   !
   if (.not.FineGd_is_expanded) then
     !
     ! k_DbGd (IBZ->BZ)
     !
     call section('=','Double Grid K-points expansion ...')
     call k_ibz2bz(FineGd_k,'i',.true.)
     call k_map_fine_to_coarse_reduce(FineGd_k)
     call msg('nsr',':: Found '//trim(intc(FineGd_k%nbz))//' ('//trim(intc(FineGd_k%nibz))//') points in the BZ (IBZ)')
     !
   endif
   !
   Xk%FGibz%FG_mode="IBZ"
   Xk%FGbz%FG_mode ="BZ"
   !
 endif
 !
 if (l_expand) then
   !
   ! BZ->IBZ
   !
   YAMBO_FREE(FineGd_k%pt)
   YAMBO_ALLOC(FineGd_k%pt,(FineGd_k%nibz,3))
   FineGd_k%nibz=FineGd_k%nbz
   FineGd_k%pt=FineGd_k%ptbz
   Xk%FGibz%FG_mode="BZ"
   !
 endif
 !
 if (l_unexpand) then
   !
   ! IBZ->BZ
   !
   FineGd_k%nbz=FineGd_k%nibz
   YAMBO_ALLOC(FineGd_k%ptbz,(FineGd_k%nbz,3))
   YAMBO_ALLOC(FineGd_k%weights,(FineGd_k%nibz))
   YAMBO_ALLOC(FineGd_k%sstar,(FineGd_k%nibz,2))
   FineGd_k%ptbz=FineGd_k%pt
   FineGd_k%weights=1./real(FineGd_k%nibz,SP)
   do i_fg=1,FineGd_k%nibz
     FineGd_k%sstar(i_fg,:)=(/i_fg,1/)
   enddo
   !
   Xk%FGibz%FG_mode="IBZ"
   Xk%FGbz%FG_mode="IBZ"
   !
 endif
 !
 ! IBZ FG_mode to IBZ map
 !========================
 call section('=','Double Grid '//trim(Xk%FGibz%FG_mode)//'-IBZ mapping')
 call k_map_fine_to_coarse(trim(Xk%FGibz%FG_mode)//'-IBZ',Xk,Xk%FGibz,FineGd_k,Xen)
 !
 ! BZ FG_mode to BZ map
 !======================
 call section('=','Double Grid '//trim(Xk%FGbz%FG_mode)//'-BZ mapping')
 call k_map_fine_to_coarse(trim(Xk%FGbz%FG_mode)//'-BZ',Xk,Xk%FGbz,FineGd_k,Xen)
 !
 ! Final I/O
 !===========
 call section('=','Double Grid Database I/O')
 !
 ! .. restoring of built-in value of nkibz to be used in io_header 
 nkibz=Xk%nibz
 !
 call io_control(ACTION=OP_WR_CL,SEC=(/1/),COM=REP,ID=ID)
 i_err=io_Double_Grid(Xen,Xk,ID,'electrons')
 !
 ! CLEAN
 !========
 call bz_samp_reset(FineGd_k)
 call k_ibz2bz(Xk,'d',.true.)
 call FineGd_E_components_reset(Xen,.TRUE.)
 call bz_samp_FineGd_reset(Xk%FGbz)
 call bz_samp_FineGd_reset(Xk%FGibz)
 !
end subroutine
