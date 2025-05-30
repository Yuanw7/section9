!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): AM
!
module R_lattice
 !
 use pars,        ONLY:SP,DP,schlen, IP
 use descriptors, ONLY:IO_desc
 !
#include<dev_defs.h>
#include<memory.h>
 !
 integer, parameter:: MAX_kpts_to_report=50
 !
 ! K lattice
 !
 integer           :: k_grid(3)
 integer           :: k_grid_dim
 real(SP)          :: k_grid_uc_vol
 real(SP)          :: k_grid_b(3,3)
 logical           :: Xk_grid_is_uniform
 logical           :: Xk_grid_new_mapping
 character(schlen) :: k_GRIDS_string="X S B C"
 integer           :: k_GRIDS_vector(4)=(/1,1,1,1/)
 !
 ! Q/K-sampling
 !
 integer  :: nqbz
 integer  :: nqibz
 integer  :: nkbz
 integer  :: nkibz
 integer  :: nXkbz
 integer  :: nXkibz
 real(SP) :: RL_vol
 real(SP) :: d3k_factor
 real(SP) :: d3q_factor
 real(SP) :: b(3,3)
 real(SP) :: q0_def_norm, q0_shift_norm, smallest_q_norm
 real(SP),allocatable :: q_norm(:)
 character(schlen)    :: q_source="is User defined and/or read from the database"
 character(schlen)    :: BZ_FineGd_mode="mixed" ! or "unexpanded"/"expanded"
 !
 type bz_fine_grid
   character(schlen)    :: FG_mode="NONE"  ! "BZ/IBZ/BARE"
   integer              :: N
   real(SP), allocatable:: pt(:,:)
   integer , allocatable:: N_mapped(:)  ! For each IBZ/BZ kpt, n° of FG k associated
   integer , allocatable:: k_range(:,:) ! For each IBZ/BZ kpt, lower/upper range of points associated in the FG
   integer,  allocatable:: E_map(:)     ! For each FG k, corresponding point in the bare FG
   integer,  allocatable:: k_map(:)     ! For each FG k, corresponding point in the IBZ/BZ
   real(SP), allocatable:: weights(:)
 end type bz_fine_grid
 !
 type(IO_desc)       :: FineGd_desc
 !
 type bz_samp
   !
   ! Coarse grid
   !
   integer           :: nibz
   integer           :: nbz
   integer,  allocatable :: nstar(:)      ! n° of points in the star
   integer,  allocatable :: star(:,:)     ! ik,ikstar --> is     is sends ik in ikstar
   integer,  allocatable :: sstar(:,:)    ! ik_bz     --> ik,is  is sends ik in ikbz
   integer,  allocatable :: s_table(:,:)  ! ik,is     --> sym@k_bz  revers of sstar(2)
   integer,  allocatable :: k_table(:,:)  ! ik,is     --> ik_bz     revers of sstar(1)
   real(SP), allocatable :: pt(:,:)      
   real(SP), allocatable :: ptbz(:,:)    
   real(SP), allocatable :: weights(:)   
   character(6)      :: description
   character(1)      :: units
   !
   ! Fine grids(s)
   !
   type (bz_fine_grid) :: FGbare
   type (bz_fine_grid) :: FGibz
   type (bz_fine_grid) :: FGbz
   !
   real(SP), allocatable :: weights_ipol(:)           ! something used by mapping with wannier
   !                                               
 end type bz_samp
 !
 type bz_map
   integer, allocatable :: k_map_inv(:,:)
   integer, allocatable :: k_map_dir(:,:,:)
   integer      :: g0_idx(3,2) ! Index of the G-vectors to bring back k-points in the BZ, in the 3 directions
   integer      :: max_kdir(3) ! Max k-points in each direction. If 1, the direction is not periodic
   real(SP)     :: g0_length(3)
   real(SP)     :: q_step(3)
 end type bz_map 
 !
 ! Q/K sampling shadow tables
 !
 real(SP),allocatable :: k_pt(:,:)
 real(SP),allocatable :: q_pt(:,:)
 integer ,allocatable :: q_sstar(:,:)
 !
 ! Q/K-tables (scattering geometry)
 !
 integer  :: QP_states_k(2)
 logical  :: bse_scattering
 logical  :: coll_scattering
 logical  :: X_scattering
 logical  :: sigma_scattering
 integer, allocatable :: qindx_X(:,:,:)
 integer, allocatable :: qindx_B(:,:,:)
 integer, allocatable :: qindx_S(:,:,:)
 integer, allocatable :: qindx_C(:,:,:)
 integer  :: qindx_B_max
 integer  :: qindx_S_max_Go
 !
 ! SYMs
 !
 real(SP),allocatable :: rl_sop(:,:,:)  ! Symmetry matrices in reciprocal space
 !
 ! RL lattice
 !
 integer              :: n_g_shells     ! Number of G-shells
 integer              :: ng_vec         ! Number of G-vectors
 integer              :: ng_closed      ! Number of G closed
 integer ,allocatable :: ng_in_shell(:) ! Number of G in each shell
 integer ,allocatable :: g_rot(:,:)
 integer ,allocatable :: G_m_G(:,:)
 integer ,allocatable :: minus_G(:)
 real(SP),allocatable :: g_vec(:,:)
 real(SP),allocatable :: E_of_shell(:)  ! Energy associated at each shell
 ! GPUs
 real(SP),allocatable DEV_ATTR :: g_vec_d(:,:)
 integer ,allocatable DEV_ATTR :: g_rot_d(:,:)
 integer ,allocatable DEV_ATTR :: G_m_G_d(:,:)
 integer ,allocatable DEV_ATTR :: minus_G_d(:)
 !
 ! RIM
 !
 integer :: RIM_ng
 integer :: RIM_id_epsm1_reference
 integer :: RIM_n_rand_pts
 logical :: RIM_is_diagonal
 real(SP):: RIM_RL_vol
 real(SP):: RIM_epsm1(3)
 real(SP):: RIM_anisotropy
 real(SP),   allocatable :: RIM_qpg(:,:,:)
 real(SP),   allocatable DEV_ATTR :: RIM_qpg_d(:,:,:)
 !
 ! RIM_W
 !
 integer :: RIM_W_ng
 integer :: RIM_W_n_rand_pts
 logical :: RIM_W_is_diagonal
 logical :: RIM_W_for_graph
 real(SP),   allocatable          :: RIM_W(:,:,:),RIM_W_E(:,:)
 real(SP),   allocatable DEV_ATTR :: RIM_W_d(:,:,:)
 real(DP),   allocatable :: f_coeff(:,:,:,:,:)
 character(schlen)       :: rimw_type
 !
 ! Coulomb (including Cutoff)
 !
 real(SP)                :: cyl_ph_radius
 real(SP)                :: cyl_length
 real(SP)                :: box_length(3)
 real(SP)                :: cyl_cut
 real(SP)                :: ws_cutoff 
 character(schlen)       :: cut_geometry
 character(schlen)       :: cut_description
 !
 complex(SP),allocatable :: bare_qpg(:,:)
 complex(SP),allocatable DEV_ATTR :: bare_qpg_d(:,:)
 !
 logical                 :: CUTOFF_plus_RIM
 real(SP)                :: cyl_vr_save
 real(SP)                :: cyl_vz_save
 real(SP)                :: cyl_zz_save
 logical                 :: cut_is_slab
 integer                 :: idir(3)
 !
 ! BZ map by direction
 !
 type(bz_map) :: k_map
 !
 ! Shift by a G-vector assoc. for points out of the BZ
 ! because of symmtry operations
 ! 
 integer, allocatable :: WF_shifts(:,:)
 !
 interface
   !
   integer function k_the_nearest(k_ref,k,n_k,k_dist,ROTATE_k_ref,k_symm,i_err,border_point)
     use pars, ONLY: SP
     integer,  intent(in) :: n_k
     real(SP), intent(in) :: k_ref(3)
     real(SP), intent(in) :: k(n_k,3)
     real(SP), intent(out),  optional :: k_dist(n_k)
     logical,  intent(out),  optional :: border_point
     logical,  intent(in) ,  optional :: ROTATE_k_ref
     integer,  intent(out),  optional :: k_symm
     integer,  intent(out),  optional :: i_err
   end function
   ! 
   function qindx_B_load(ikbz,ikp_bz,ID)
     !
     integer, dimension(2) :: qindx_B_load
     integer, intent(in) :: ID,ikbz,ikp_bz
     !
   end function qindx_B_load
   !
 end interface
 !
 contains
   !
   subroutine qindx_alloc(what)
     !
     use parallel_m,   ONLY:PAR_K_scheme
     !
     character(*), intent(in) :: what
     !
     integer :: nK_qindx_B
     !
     if(  (index(what,"X")/=0 .or. index(what,"ALL")/=0) .and. X_scattering) then
       YAMBO_ALLOC(qindx_X,(nqibz,nXkbz,2))
       qindx_X=0
     endif
     !
     if( (index(what,"S")/=0 .or. index(what,"ALL")/=0) .and. Xk_grid_is_uniform .and. sigma_scattering) then
       YAMBO_ALLOC(qindx_S,(QP_states_k(2),nqbz,2))
       qindx_S=0
     endif
     !
     if( (index(what,"B")/=0 .or. index(what,"ALL")/=0) .and. Xk_grid_is_uniform .and. bse_scattering) then
#if defined _PAR_IO
       nK_qindx_B=PAR_K_scheme%nbz
#else
       if(PAR_K_scheme%nbz> 0) nK_qindx_B=PAR_K_scheme%nbz  ! read mode
       if(PAR_K_scheme%nbz==0) nK_qindx_B=nXkbz             ! write mode
#endif
       YAMBO_ALLOC(qindx_B,(nXkbz,nK_qindx_B,2))
       qindx_B=0
     endif
     !
     if( (index(what,"C")/=0 .or. index(what,"ALL")/=0) .and. Xk_grid_is_uniform .and. coll_scattering) then
       YAMBO_ALLOC(qindx_C,(nXkbz,nXkbz,2))
       qindx_C=0
     endif
     !
   end subroutine
   !
   subroutine qindx_free(what)
     !
     character(*), intent(in) :: what
     !
     if(  index(what,"X")/=0 .or. index(what,"ALL")/=0) then
       YAMBO_FREE(qindx_X)
     endif
     if(  index(what,"B")/=0 .or. index(what,"ALL")/=0) then
       YAMBO_FREE(qindx_B)
     endif
     if(  index(what,"S")/=0 .or. index(what,"ALL")/=0) then
       YAMBO_FREE(qindx_S)
     endif
     if(  index(what,"C")/=0 .or. index(what,"ALL")/=0) then
       YAMBO_FREE(qindx_C)
     endif
   end subroutine qindx_free
   !
   subroutine bz_samp_FineGd_alloc(FG,N_coarse)
     type(bz_fine_grid)::FG
     integer           ::N_coarse,N_alloc
     !
     if (allocated( FG%k_range )) return
     !
     if (FG%N>0 ) N_alloc=FG%N
     if (FG%N==0) N_alloc=N_coarse
     !
     YAMBO_ALLOC(FG%pt,(N_alloc,3))
     YAMBO_ALLOC(FG%weights,(N_alloc))
     YAMBO_ALLOC(FG%E_map,(N_alloc))
     YAMBO_ALLOC(FG%k_map,(N_alloc))
     FG%pt=0._SP
     FG%weights=0._SP
     FG%E_map=0
     FG%k_map=0
     YAMBO_ALLOC(FG%k_range,(N_coarse,2))
     YAMBO_ALLOC(FG%N_mapped,(N_coarse))
     FG%k_range=0
     FG%N_mapped=0
   end subroutine
   !
   subroutine bz_samp_FineGd_default(FG,k,nk)
     type(bz_fine_grid)::FG
     type(bz_samp)     ::k
     integer, optional ::nk 
     ! ws
     integer           ::ik,ic,nk_
     nk_=k%nibz
     if (present(nk)) then
       nk_=nk
     endif
     FG%N=nk_
     do ik=1,nk_
       if (nk_>k%nibz) then
         FG%pt(ik,:)=k%ptbz(ik,:)
         FG%weights(ik)=1._SP
       else
         FG%pt(ik,:)=k%pt(ik,:)
         FG%weights(ik)=k%weights(ik)
       endif
       FG%k_range(ik,:)=ik
       FG%E_map(ik)=ik
       FG%k_map(ik)=ik
       FG%N_mapped(ik)=1
     enddo
   end subroutine
   !
   subroutine bz_samp_FineGd_reset(FG)
     type(bz_fine_grid)::FG
     FG%N=0
     FG%FG_mode="NONE"
     YAMBO_FREE(FG%pt)
     YAMBO_FREE(FG%weights)
     YAMBO_FREE(FG%E_map)
     YAMBO_FREE(FG%k_range)
     YAMBO_FREE(FG%N_mapped)
   end subroutine
   !
   subroutine bz_samp_reset(k)
     type(bz_samp)::k
     k%nibz=0
     k%nbz=0
     k%units=' '
     k%description=' '
     !
     YAMBO_FREE(k%s_table)
     YAMBO_FREE(k%k_table)
     YAMBO_FREE(k%nstar)
     YAMBO_FREE(k%star)
     YAMBO_FREE(k%sstar)
     YAMBO_FREE(k%pt)
     YAMBO_FREE(k%ptbz)
     YAMBO_FREE(k%weights)
     !
     ! Double Grid
     !
     k%FGbz%N=0
     k%FGibz%N=0
     k%FGbare%N=0
     !
     call bz_samp_FineGd_reset(k%FGbz)
     call bz_samp_FineGd_reset(k%FGibz)
     call bz_samp_FineGd_reset(k%FGbare)
     !
   end subroutine bz_samp_reset
   !
   subroutine bz_samp_duplicate(BZi, BZo)
     type(bz_samp), intent(in)    :: BZi
     type(bz_samp), intent(inout) :: BZo
     !
     BZo%nibz  = BZi%nibz
     BZo%nbz   = BZi%nbz
     BZo%units = BZi%units
     BZo%description = BZi%description
     YAMBO_ALLOC(BZo%pt,(BZo%nibz,3))
     BZo%pt(:,:) = BZi%pt(:,:)
     if (allocated( BZi%nstar )) then
       YAMBO_ALLOC(BZo%nstar,(BZo%nibz))
       BZo%nstar=BZi%nstar
     endif
     if (allocated( BZi%weights )) then
       YAMBO_ALLOC(BZo%weights,(BZo%nibz))
       BZo%weights=BZi%weights
     endif
     if (allocated( BZi%star )) then
       YAMBO_ALLOC(BZo%star,(BZo%nibz,size(BZi%star,2)))
       BZo%star=BZi%star
     endif
     if (allocated( BZi%sstar )) then
       YAMBO_ALLOC(BZo%sstar,(BZo%nbz,2))
       BZo%sstar=BZi%sstar
     endif
     if (allocated( BZi%ptbz )) then
       YAMBO_ALLOC(BZo%ptbz,(BZo%nbz,3))
       BZo%ptbz=BZi%ptbz
     endif
     !
   end subroutine bz_samp_duplicate
     !
   subroutine bz_samp_duplicate_Fg(FGi,FGo,N_coarse)
     type(bz_fine_grid), intent(in)  ::FGi
     type(bz_fine_grid), intent(out) ::FGo
     integer,            intent(in)  ::N_coarse
     !
     FGo%N =FGi%N
     FGo%FG_mode=FGi%FG_mode
     !
     if (allocated(FGi%pt)) then
       YAMBO_ALLOC(FGo%pt,(FGo%N,3))
       FGo%pt=FGi%pt
     endif
     if (allocated(FGi%weights)) then
       YAMBO_ALLOC(FGo%weights,(FGo%N))
       FGo%weights=FGi%weights
     endif
     if (allocated(FGi%k_map)) then
       YAMBO_ALLOC(FGo%k_map,(FGo%N))
       FGo%k_map=FGi%k_map
     endif
     if (allocated(FGi%E_map)) then
       YAMBO_ALLOC(FGo%E_map,(FGo%N))
       FGo%E_map=FGi%E_map
     endif
     if (allocated(FGi%k_range)) then
       YAMBO_ALLOC(FGo%k_range,(N_coarse,2))
       FGo%k_range=FGi%k_range
     endif
     if (allocated(FGi%N_mapped)) then
       YAMBO_ALLOC(FGo%N_mapped,(N_coarse))
       FGo%N_mapped=FGi%N_mapped
     endif
     !
     !DEBUG>
     !To code 
     ! real(SP), pointer :: weights_ipol(:)   => null()  ! something used by mapping with wannier
     !DEBUG<
     !
     return
     !
   end subroutine bz_samp_duplicate_Fg
   !
   subroutine cutoff_presets()
     Box_length=0.
     cyl_ph_radius=0.
     box_length=0.
     cyl_length=0.
     ws_cutoff=0.7
     cut_geometry='none'
     cut_description='none'
     CUTOFF_plus_RIM=.false.
   end subroutine
   !
end module R_lattice
