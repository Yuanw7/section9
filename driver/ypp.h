/*
  Copyright (C) 2000-2012 A. Marini and the YAMBO team 
               http://www.yambo-code.org
  
  This file is distributed under the terms of the GNU 
  General Public License. You can redistribute it and/or 
  modify it under the terms of the GNU General Public 
  License as published by the Free Software Foundation; 
  either version 2, or (at your option) any later version.
 
  This program is distributed in the hope that it will 
  be useful, but WITHOUT ANY WARRANTY; without even the 
  implied warranty of MERCHANTABILITY or FITNESS FOR A 
  PARTICULAR PURPOSE.  See the GNU General Public License 
  for more details.
 
  You should have received a copy of the GNU General Public 
  License along with this program; if not, write to the Free 
  Software Foundation, Inc., 59 Temple Place - Suite 330,Boston, 
  MA 02111-1307, USA or visit http://www.gnu.org/copyleft/gpl.txt.
*/
/*
  declaration
*/
/*
 "e" and "s" commmand line structure
*/
#if defined _FORTRAN_US
 int ypp_i
#else
 int ypp_i_
#endif
(int *, int *,int *,int *,int *,int *,int *,int *,
  char *rnstr2, char *inf, char *id, char *od, char *com_dir, char *js,
  int lni,int iif,int iid,int iod,int icd,int ijs);
/*
 Command line structure
*/
 static Ldes opts[] = { /* Int Real Ch (dummy) Parallel_option*/
  {"help",  "h","Short Help",0,0,0,0,0}, 
  {"lhelp", "H","Long Help",0,0,0,0,0}, 
  {"jobstr","J","Job string identifier",0,0,1,0,1},   
  {"infver","V","Input file verbosity [opt=gen,qp,all]",0,0,1,0,0},    
  {"ifile", "F","Input file",0,0,1,0,1},              
  {"idir",  "I","Core I/O directory",0,0,1,0,1},         
  {"odir",  "O","Additional I/O directory",0,0,1,0,1},        
  {"cdir",  "C","Communications I/O directory",0,0,1,0,1},
  {"nompi", "N","Skip MPI initialization",0,0,0,0,0}, 
  {"dbfrag","S","DataBases fragmentation",0,0,0,0,1}, 
  {"bzgrids","k","BZ Grid generator [(k)pt,(q)pt,(l)ongitudinal,(h)igh symmetry]",0,0,1,0,0}, 
  {"excitons", "e","Excitons  [(s)ort,(a)mp,(w)ave]",0,0,1,0,0}, 
  {"qpdb",   "q","Generate/modify a quasi-particle database",0,0,0,0,0}, 
  {"electrons","s","Electrons [(w)ave,(d)ensity,do(s)]",0,0,1,0,0}, 
  {"freehole","f","Free hole position [excitons plot]",0,0,0,0,0}, 
  {"bzrim",   "r","BZ energy RIM analyzer",0,0,0,0,0}, 
  {NULL,NULL,NULL,0,0,0,0,0}
 };
 char *tool="ypp";
 char *tdesc="Y(ambo) P(ost) P(rocessor)";
