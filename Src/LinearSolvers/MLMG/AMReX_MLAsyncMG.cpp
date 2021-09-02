#include <AMReX_MLAsyncMG.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_MLMG_K.H>
#include <AMReX_MLABecLaplacian.H>

#ifdef AMREX_USE_PETSC
#include <petscksp.h>
#include <AMReX_PETSc.H>
#endif

#ifdef AMREX_USE_EB
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_MLEBABecLap.H>
#endif

namespace amrex {

MLAsyncMG::MLAsyncMG (MLLinOp& a_lp) : MLMG(a_lp)
{}

MLAsyncMG::~MLAsyncMG ()
{}

Real
MLAsyncMG::asyncSolve (const Vector<MultiFab*>& a_sol, const Vector<MultiFab const*>& a_rhs,
                       Real a_tol_rel, Real a_tol_abs, const char* checkpoint_file)
{
    BL_PROFILE("MLAsyncMG::solve()");

    if (bottom_solver == BottomSolver::Default) {
        bottom_solver = linop.getDefaultBottomSolver();
    }

    if (bottom_solver == BottomSolver::hypre || bottom_solver == BottomSolver::petsc) {
        int mo = linop.getMaxOrder();
        if (a_sol[0]->hasEBFabFactory()) {
            linop.setMaxOrder(2);
        } else {
            linop.setMaxOrder(std::min(3,mo));  // maxorder = 4 not supported
        }
    }

    bool is_nsolve = linop.m_parent;

    auto solve_start_time = amrex::second();

    Real& composite_norminf = m_final_resnorm0;

    m_niters_cg.clear();
    m_iter_fine_resnorm0.clear();

    prepareForAsyncSolve(a_sol, a_rhs);

    computeMLResidual(finest_amr_lev);

    int ncomp = linop.getNComp();

    (*cor_smooth[0]).setVal(0.0);
    MultiFab::Copy(*res_finest[0], res[0][0], 0, 0, ncomp, 0);

    bool local = true;
    Real resnorm0 = MLResNormInf(finest_amr_lev, local);
    Real rhsnorm0 = MLRhsNormInf(local);
    if (!is_nsolve) {
        ParallelAllReduce::Max<Real>({resnorm0, rhsnorm0}, ParallelContext::CommunicatorSub());

        if (verbose >= 1)
        {
            amrex::Print() << "MLAsyncMG: Initial rhs               = " << rhsnorm0 << "\n"
                           << "MLAsyncMG: Initial residual (resid0) = " << resnorm0 << "\n";
        }
    }

    m_init_resnorm0 = resnorm0;
    m_rhsnorm0 = rhsnorm0;

    Real max_norm;
    std::string norm_name;
    if (always_use_bnorm || rhsnorm0 >= resnorm0) {
        norm_name = "bnorm";
        max_norm = rhsnorm0;
    } else {
        norm_name = "resid0";
        max_norm = resnorm0;
    }
    const Real res_target = std::max(a_tol_abs, std::max(a_tol_rel,Real(1.e-16))*max_norm);

    if (!is_nsolve && resnorm0 <= res_target) {
        composite_norminf = resnorm0;
        if (verbose >= 1) {
            amrex::Print() << "MLAsyncMG: No iterations needed\n";
        }
    } else {
        MLCellLinOp *cell_linop = dynamic_cast<MLCellLinOp *>(&linop);
	MultiFab& cor_smooth_alev = *cor_smooth[0];
        cor_smooth_alev.AsyncSetup();

        auto iter_start_time = amrex::second();
        bool converged = false;

        const int niters = do_fixed_number_of_iters ? do_fixed_number_of_iters : max_iters;

        

        for (int iter = 0; iter < niters; ++iter)
        {
            asyncOneIter(iter);

            converged = false;

	    MultiFab::Copy(*cor_finest[0], cor_smooth_alev, 0, 0, ncomp, 0);
            testComputeResidual(finest_amr_lev);
	    //computeResidual(finest_amr_lev);

            if (is_nsolve) continue;

            Real fine_norminf = ResNormInf(finest_amr_lev);
            //{
            //    bool comm_complete;
	    //    testInit(comm_complete);
            //    while (!comm_complete)
            //    {
            //        fine_norminf = testResNormInf(finest_amr_lev, comm_complete);
            //    }
            //}

            m_iter_fine_resnorm0.push_back(fine_norminf);
            composite_norminf = fine_norminf;
            if (verbose >= 2) {
                amrex::Print() << "MLAsyncMG: Iteration " << std::setw(3) << iter+1 << " Fine resid/"
                               << norm_name << " = " << fine_norminf/max_norm << "\n";
            }
            bool fine_converged = (fine_norminf <= res_target);

            //if (namrlevs == 1 && fine_converged) {
            //    converged = true;
            //} else if (fine_converged) {
            //    // finest level is converged, but we still need to test the coarse levels
            //    computeMLResidual(finest_amr_lev-1);
            //    Real crse_norminf = MLResNormInf(finest_amr_lev-1);
            //    if (verbose >= 2) {
            //        amrex::Print() << "MLAsyncMG: Iteration " << std::setw(3) << iter+1
            //                       << " Crse resid/" << norm_name << " = "
            //                       << crse_norminf/max_norm << "\n";
            //    }
            //    converged = (crse_norminf <= res_target);
            //    composite_norminf = std::max(fine_norminf, crse_norminf);
            //} else {
            //    converged = false;
            //}

	    if (fine_converged)
	    {
                converged = true;
	    }
	    else
	    {
                converged = false;
	    }

            if (converged) {
                if (verbose >= 1) {
                    amrex::Print() << "MLAsyncMG: Final Iter. " << iter+1
                                   << " resid, resid/" << norm_name << " = "
                                   << composite_norminf << ", "
                                   << composite_norminf/max_norm << "\n";
                }
                break;
            } else {
              if (composite_norminf > Real(1.e20)*max_norm)
              {
                  if (verbose > 0) {
                      amrex::Print() << "MLAsyncMG: Failing to converge after " << iter+1 << " iterations."
                                     << " resid, resid/" << norm_name << " = "
                                     << composite_norminf << ", "
                                     << composite_norminf/max_norm << "\n";
                  }
                  amrex::Abort("MLAsyncMG failing so lets stop here");
              }
            }
        }

        if (!converged && do_fixed_number_of_iters == 0) {
            if (verbose > 0) {
                amrex::Print() << "MLAsyncMG: Failed to converge after " << max_iters << " iterations."
                               << " resid, resid/" << norm_name << " = "
                               << composite_norminf << ", "
                               << composite_norminf/max_norm << "\n";
            }
            amrex::Abort("MLAsyncMG failed");
        }

	const int cross = cell_linop->isCrossStencil();
        const Geometry& m_geom = cell_linop->Geom(0, 0);
	int sweep = 0;
        while (1)
        {
            asyncSmoothSweep(&linop, 0, 0, cor_smooth_alev, *res_finest[0], true);
            sweep++;
            int converge_flag = cor_smooth_alev.AsyncCheckConverge(1, 0);
            if (converge_flag == 1){
               break;
            }
        }

        cor_smooth_alev.AsyncCleanup(0, ncomp, cor_smooth_alev.nGrowVect(), m_geom.periodicity(), cross);

	MultiFab::Add(*sol[0], cor_smooth_alev, 0, 0, ncomp, 0);

        timer[iter_time] = amrex::second() - iter_start_time;
    }

    IntVect ng_back = final_fill_bc ? IntVect(1) : IntVect(0);
    if (linop.hasHiddenDimension()) {
        ng_back[linop.hiddenDirection()] = 0;
    }
    for (int alev = 0; alev < namrlevs; ++alev)
    {
        if (a_sol[alev] != sol[alev])
        {
            MultiFab::Copy(*a_sol[alev], *sol[alev], 0, 0, ncomp, ng_back);
        }
    }

    timer[solve_time] = amrex::second() - solve_start_time;
    if (verbose >= 1) {
        ParallelReduce::Max<double>(timer.data(), timer.size(), 0,
                                    ParallelContext::CommunicatorSub());
        if (ParallelContext::MyProcSub() == 0)
        {
            amrex::AllPrint() << "MLMG: Timers: Solve = " << timer[solve_time]
                              << " Iter = " << timer[iter_time]
                              << " Bottom = " << timer[bottom_time] << "\n";
        }
    }

    ++solve_called;

    return composite_norminf;
}

Real
MLAsyncMG::testResNormInf (int alev, bool& comm_complete, bool local)
{
    BL_PROFILE("MLMG::ResNormInf()");
    const int ncomp = linop.getNComp();
    const int mglev = 0;
    Real norm = 0.0;
    MultiFab* pmf = &(res[alev][mglev]);
//#ifdef AMREX_USE_EB
//    if (linop.isCellCentered() && scratch[alev]) {
//        pmf = scratch[alev].get();
//        MultiFab::Copy(*pmf, res[alev][mglev], 0, 0, ncomp, 0);
//        auto factory = dynamic_cast<EBFArrayBoxFactory const*>(linop.Factory(alev));
//        if (factory) {
//            const MultiFab& vfrac = factory->getVolFrac();
//            for (int n=0; n < ncomp; ++n) {
//                MultiFab::Multiply(*pmf, vfrac, 0, n, 1, 0);
//            }
//        } else {
//            amrex::Abort("MLMG::ResNormInf: not EB Factory");
//        }
//    }
//#endif
    if (!local)
    {
        if (test_count == 0)
        {
            for (int n = 0; n < ncomp; n++)
            {
                Real newnorm = 0.0;
                if (fine_mask[alev]) {
                    newnorm = pmf->norm0(*fine_mask[alev],n,0,true);
                } else {
                    newnorm = pmf->norm0(n,0,true);
                }
                norm = std::max(norm, newnorm);
            }
	    reduce_req = MPI_REQUEST_NULL;
            reduce_recv = new Real[1];
            reduce_send = new Real[1];
	    reduce_send[0] = norm;
	    MPI_Iallreduce(reduce_send, reduce_recv, 1, MPI_DOUBLE, MPI_MAX, ParallelContext::CommunicatorSub(), &reduce_req);
	}

	int reduce_flag = 0;
        ParallelDescriptor::Test(reduce_req, reduce_flag, reduce_stat);
        if (reduce_flag)
        {
            comm_complete = true;
            norm = reduce_recv[0];
            delete[] reduce_recv;
            delete[] reduce_send;
            test_count++;
        }
        else
        {
            comm_complete = false;
            test_count++;
            return 0;
        }
    }
    else
    {
        for (int n = 0; n < ncomp; n++)
        {
            Real newnorm = 0.0;
            if (fine_mask[alev]) {
                newnorm = pmf->norm0(*fine_mask[alev],n,0,true);
            } else {
                newnorm = pmf->norm0(n,0,true);
            }
            norm = std::max(norm, newnorm);
        }
    }

    return norm;
}

void
MLAsyncMG::testComputeResidual (int alev)
{
    BL_PROFILE("MLMG::computeResidual()");

    //MultiFab& x = *sol[alev];
    //const MultiFab& b = rhs[alev];
    //MultiFab& r = res[alev][0];

    //const MultiFab* crse_bcdata = nullptr;
    ////if (alev > 0) {
    ////    crse_bcdata = sol[alev-1];
    ////}

    //MLCellLinOp *cell_linop = dynamic_cast<MLCellLinOp *>(&linop);
    //bool comm_complete;
    //cell_linop->testInit(comm_complete);
    //while (!comm_complete)
    //{
    //    cell_linop->testSolutionResidual(alev, r, x, b, comm_complete, crse_bcdata);
    //}

    MultiFab& x = *cor_finest[alev];
    const MultiFab& b = *res_finest[alev];
    MultiFab& r = res[alev][0];

    MLCellLinOp *cell_linop = dynamic_cast<MLCellLinOp *>(&linop);
    bool comm_complete;
    cell_linop->testInit(comm_complete);
    while (!comm_complete)
    {
	cell_linop->testCorrectionResidual(alev, 0, r, x, b, BCMode::Homogeneous, comm_complete);
    }
}

void
MLAsyncMG::testComputeResOfCorrection (int amrlev, int mglev)
{
    BL_PROFILE("MLMG:computeResOfCorrection()");
    MultiFab& x = *cor[amrlev][mglev];
    const MultiFab& b = res[amrlev][mglev];
    MultiFab& r = rescor[amrlev][mglev];

    MLCellLinOp *cell_linop = dynamic_cast<MLCellLinOp *>(&linop);
    bool comm_complete;
    cell_linop->testInit(comm_complete);
    while (!comm_complete)
    {
        cell_linop->testCorrectionResidual(amrlev, mglev, r, x, b, BCMode::Homogeneous, comm_complete);
    }
}

// in  : Residual (res) on the finest AMR level
// out : sol on all AMR levels
void MLAsyncMG::asyncOneIter (int iter)
{
    BL_PROFILE("MLMG::oneIter()");

    int ncomp = linop.getNComp();
    //int nghost = 0;
    //if (cf_strategy == CFStrategy::ghostnodes) nghost = linop.getNGrow();

    //for (int alev = finest_amr_lev; alev > 0; --alev)
    //{
    //    mgAddVcycle(alev, 0);

    //    MultiFab::Add(*sol[alev], *cor[alev][0], 0, 0, ncomp, nghost);

    //    // compute residual for the coarse AMR level
    //    computeResWithCrseSolFineCor(alev-1,alev);

    //    if (alev != finest_amr_lev) {
    //        std::swap(cor_hold[alev][0], cor[alev][0]); // save it for the up cycle
    //    }
    //}

    // coarsest amr level
    {
        //// enforce solvability if appropriate
        //if (linop.isSingular(0) && linop.getEnforceSingularSolvable())
        //{
        //    makeSolvable(0,0,res[0][0]);
        //}
       
        mgAddVcycle(0, 0);

        //MultiFab::Add(*sol[0], *cor[0][0], 0, 0, ncomp, 0);
    }

    //for (int alev = 1; alev <= finest_amr_lev; ++alev)
    //{
    //    // (Fine AMR correction) = I(Coarse AMR correction)
    //    interpCorrection(alev);

    //    MultiFab::Add(*sol[alev], *cor[alev][0], 0, 0, ncomp, nghost);

    //    if (alev != finest_amr_lev) {
    //        MultiFab::Add(*cor_hold[alev][0], *cor[alev][0], 0, 0, ncomp, nghost);
    //    }

    //    // Update fine AMR level correction
    //    computeResWithCrseCorFineCor(alev);

    //    mgAddVcycle(alev, 0);

    //    MultiFab::Add(*sol[alev], *cor[alev][0], 0, 0, ncomp, nghost);

    //    if (alev != finest_amr_lev) {
    //        MultiFab::Add(*cor[alev][0], *cor_hold[alev][0], 0, 0, ncomp, nghost);
    //    }
    //}

    //averageDownAndSync();
}

//Real
//MLMG::testResNormInf (int alev, bool local)
//{
//    BL_PROFILE("MLMG::ResNormInf()");
//    const int ncomp = linop.getNComp();
//    const int mglev = 0;
//    Real norm = 0.0;
//    MultiFab* pmf = &(res[alev][mglev]);
//#ifdef AMREX_USE_EB
//    if (linop.isCellCentered() && scratch[alev]) {
//        pmf = scratch[alev].get();
//        MultiFab::Copy(*pmf, res[alev][mglev], 0, 0, ncomp, 0);
//        auto factory = dynamic_cast<EBFArrayBoxFactory const*>(linop.Factory(alev));
//        if (factory) {
//            const MultiFab& vfrac = factory->getVolFrac();
//            for (int n=0; n < ncomp; ++n) {
//                MultiFab::Multiply(*pmf, vfrac, 0, n, 1, 0);
//            }
//        } else {
//            amrex::Abort("MLMG::ResNormInf: not EB Factory");
//        }
//    }
//#endif
//    for (int n = 0; n < ncomp; n++)
//    {
//        Real newnorm = 0.0;
//        if (fine_mask[alev]) {
//            newnorm = pmf->norm0(*fine_mask[alev],n,0,true);
//        } else {
//            newnorm = pmf->norm0(n,0,true);
//        }
//        norm = std::max(norm, newnorm);
//    }
//    if (!local)
//    {
//        ParallelAllReduce::Max(norm, ParallelContext::CommunicatorSub());
//    }
//    return norm;
//}

void
MLAsyncMG::prepareForAsyncSolve (const Vector<MultiFab*>& a_sol, const Vector<MultiFab const*>& a_rhs)
{
    prepareForSolve(a_sol, a_rhs);

    const int ncomp = linop.getNComp();
    IntVect ng_sol(1);
    if (linop.hasHiddenDimension()) ng_sol[linop.hiddenDirection()] = 0;

    IntVect ng = linop.isCellCentered() ? IntVect(0) : IntVect(1);
    if (cf_strategy != CFStrategy::ghostnodes) ng = ng_sol;

    cor_finest.resize(namrlevs);
    for (int alev = 0; alev <= finest_amr_lev; ++alev)
    {
        cor_finest[alev] = std::make_unique<MultiFab>(res[alev][0].boxArray(),
                                                      res[alev][0].DistributionMap(),
                                                      ncomp, ng, MFInfo(),
                                                      *linop.Factory(alev,0));
    }

    res_finest.resize(namrlevs);
    for (int alev = 0; alev <= finest_amr_lev; ++alev)
    {
        res_finest[alev] = std::make_unique<MultiFab>(res[alev][0].boxArray(),
                                                      res[alev][0].DistributionMap(),
                                                      ncomp, ng, MFInfo(),
                                                      *linop.Factory(alev,0));
    }

    cor_smooth.resize(namrlevs);
    for (int alev = 0; alev <= finest_amr_lev; ++alev)
    {
        cor_smooth[alev] = std::make_unique<MultiFab>(res[alev][0].boxArray(),
                                                      res[alev][0].DistributionMap(),
                                                      ncomp, ng, MFInfo(),
                                                      *linop.Factory(alev,0));
    }

    Acor.resize(namrlevs);
    for (int alev = 0; alev <= finest_amr_lev; ++alev)
    {
        Acor[alev] = std::make_unique<MultiFab>(res[alev][0].boxArray(),
                                                res[alev][0].DistributionMap(),
                                                ncomp, ng, MFInfo(),
                                                *linop.Factory(alev,0));
    }

    asyncFillBoundary_count = 0;
}

void
MLAsyncMG::mgAddVcycle (int amrlev, int mglev_top)
{
    BL_PROFILE("MLMG::oneIter()");

    const int ncomp = linop.getNComp();
    const int mglev_bottom = linop.NMGLevels(amrlev) - 1;
    bool skip_fillboundary;
    int nghost = 0;
    if (cf_strategy == CFStrategy::ghostnodes) nghost = linop.getNGrow();

    MultiFab& cor_top = *cor[amrlev][mglev_top];
    MultiFab& res_top = res[amrlev][mglev_top];
    MLCellLinOp *cell_linop = dynamic_cast<MLCellLinOp *>(&linop);

    cor_top.setVal(0.0);

    if (linop.NMGLevels(amrlev) < 1){
       return;
    }
    else if (amrlev != 0){
       int num_smooth_sweeps = 1;
       skip_fillboundary = true;
       for (int i = 0; i < num_smooth_sweeps; ++i) {
           linop.smooth(amrlev, mglev_top, cor_top, res_top, skip_fillboundary);
           skip_fillboundary = false;
       }
    }
    else {
       MultiFab& cor_smooth_alev = *cor_smooth[amrlev];


       if (0){
          /* Restrict finest residual to all coarse grids */
          for (int mglev = mglev_top; mglev < mglev_bottom; mglev++)
          {
              linop.restriction(amrlev, mglev+1, res[amrlev][mglev+1], res[amrlev][mglev]);
          }
          /* Concurrent smooth and and bottom solve*/
          for (int mglev = mglev_top; mglev <= mglev_bottom; mglev++)
          {
             if (mglev == mglev_bottom)
             {
                 bottomSolve();
             }
             else if (mglev == mglev_top)
             {
                 cor[amrlev][mglev]->setVal(0.0);
             }
             else
             {
                 cor[amrlev][mglev]->setVal(0.0);
                 skip_fillboundary = true;
                 int num_smooth_sweeps = 1;
                 for (int i = 0; i < num_smooth_sweeps; ++i) {
                     linop.smooth(amrlev, mglev, *cor[amrlev][mglev], res[amrlev][mglev],
                                  skip_fillboundary);
                     skip_fillboundary = false;
                 }
             }   
          }
          /* Sum up corrections using interpolation */
          for (int mglev = mglev_bottom-1; mglev >= mglev_top; mglev--)
          {
              addInterpCorrection(amrlev, mglev);
          }
       }
       else {
          linop.restriction(amrlev, mglev_top+1, res[amrlev][mglev_top+1], res[amrlev][mglev_top]);
 
          int inner_niters = 1;
          for (int inner_iter = 0; inner_iter < inner_niters; inner_iter++){
             for (int mglev = mglev_top+1; mglev < mglev_bottom; ++mglev)
             {
                if (mglev == mglev_top+1 && inner_iter > 0){
                   skip_fillboundary = false;
                }
                else
		{
                   cor[amrlev][mglev]->setVal(0.0);
                   skip_fillboundary = true;
                }

                //for (int i = 0; i < nu1; ++i)
                //{
                //    linop.smooth(amrlev, mglev, *cor[amrlev][mglev], res[amrlev][mglev],
                //                 skip_fillboundary);
                //    skip_fillboundary = false;
                //}

                syncSmooth(amrlev, mglev, nu1, skip_fillboundary);
                testComputeResOfCorrection(amrlev, mglev);
                linop.restriction(amrlev, mglev+1, res[amrlev][mglev+1], rescor[amrlev][mglev]);
             }

	     {
                 int mglev = mglev_top;

	         //skip_fillboundary = true;
                 //for (int i = 0; i < 10; ++i)
                 //{
                 //    linop.smooth(amrlev, mglev, *cor[amrlev][mglev], res[amrlev][mglev],
                 //                 skip_fillboundary);
                 //    skip_fillboundary = false;
                 //}

		 skip_fillboundary = true;
                 syncSmooth(amrlev, mglev, nu1, skip_fillboundary);
	     }

             //bottomSolve();


             for (int mglev = mglev_bottom-1; mglev >= mglev_top+1; --mglev)
             {
                 addInterpCorrection(amrlev, mglev);

                 //skip_fillboundary = false;
                 //for (int i = 0; i < nu2; ++i)
                 //{
                 //    linop.smooth(amrlev, mglev, *cor[amrlev][mglev], res[amrlev][mglev],
                 //                 skip_fillboundary);
                 //}

		 skip_fillboundary = false;
                 syncSmooth(amrlev, mglev, nu1, skip_fillboundary);
             }
          }

          cor[amrlev][mglev_top]->setVal(0.0);
          addInterpCorrection(amrlev, mglev_top);
       }

       //if (0) { 
          //cor_smooth_alev.setVal(0.0);
          //skip_fillboundary = true;
          //for (int i = 0; i < num_top_async_smooth_sweeps; ++i) {
          //    linop.smooth(amrlev, mglev_top, cor_smooth_alev, res_top, skip_fillboundary);
          //    skip_fillboundary = false;
          //}
          //skip_fillboundary = false;
          //for (int i = 0; i < num_top_async_smooth_sweeps; ++i) {
          //    linop.smooth(amrlev, mglev_top, *sol[amrlev], rhs[amrlev], skip_fillboundary);
          //}

          int num_top_async_smooth_sweeps;
	  num_top_async_smooth_sweeps = 1;
	  skip_fillboundary = false;
          for (int i = 0; i < num_top_async_smooth_sweeps; ++i) {
	      asyncSmoothSweep(&linop, 0, 0, cor_smooth_alev, *res_finest[0], true);
          }
 
          MultiFab& Acor_alev = *Acor[amrlev];
          MultiFab& rescor_top = rescor[amrlev][mglev_top];

	  //cell_linop->solutionResidual(amrlev, rescor_top, *sol[amrlev], rhs[amrlev], nullptr);
	  //cell_linop->correctionResidual(amrlev, mglev_top, rescor_top, cor_smooth_alev, res_top, BCMode::Homogeneous);
	  //cell_linop->correctionResidual(amrlev, mglev_top, rescor_top, cor_smooth_alev, *res_finest[0], BCMode::Homogeneous);

	  (*cor_finest[0]).setVal(0.0);
          MultiFab::Add(*cor_finest[0], cor_smooth_alev, 0, 0, ncomp, nghost);
	  {
	      bool comm_complete;
              cell_linop->testInit(comm_complete);
	      int sweeps = 0;
              while (!comm_complete)
              {
                  //cell_linop->testSolutionResidual(amrlev, rescor_top, *sol[amrlev], rhs[amrlev], comm_complete, nullptr);
	          cell_linop->testCorrectionResidual(amrlev, mglev_top, rescor_top, *cor_finest[0], *res_finest[0], BCMode::Homogeneous, comm_complete);
		  asyncSmoothSweep(&linop, 0, 0, cor_smooth_alev, *res_finest[0], true);
		  sweeps++;
	      }
	      //int rank;
	      //MPI_Comm_rank(ParallelContext::CommunicatorSub(), &rank);
	      //printf("%d %d\n", rank, sweeps);
	      //fflush(stdout);
	  }

          //Real beta_numer = linop.xdoty(amrlev, mglev_top, cor_top, rescor_top, false);
	  Real beta_numer;
	  {
	      bool comm_complete;
              cell_linop->testInit(comm_complete);
              while (!comm_complete)
              {
                  beta_numer = cell_linop->testXdoty(amrlev, mglev_top, cor_top, rescor_top, false, comm_complete);
		  asyncSmoothSweep(&linop, 0, 0, cor_smooth_alev, *res_finest[0], true);
	      }
	  }

	  //cell_linop->apply(amrlev, mglev_top, Acor_alev, cor_top, MLLinOp::BCMode::Homogeneous, MLLinOp::StateMode::Correction);
	  {
              bool comm_complete;
              cell_linop->testInit(comm_complete);
              while (!comm_complete)
              {
                  cell_linop->testApply(amrlev, mglev_top, Acor_alev, cor_top, MLLinOp::BCMode::Homogeneous, MLLinOp::StateMode::Correction, comm_complete);
		  asyncSmoothSweep(&linop, 0, 0, cor_smooth_alev, *res_finest[0], true);
              }
          }

          //Real beta_denom = linop.xdoty(amrlev, mglev_top, cor_top, Acor_alev, false);
	  Real beta_denom;
          {
              bool comm_complete;
              cell_linop->testInit(comm_complete);
              while (!comm_complete)
              {
                  beta_denom = cell_linop->testXdoty(amrlev, mglev_top, cor_top, Acor_alev, false, comm_complete);
		  asyncSmoothSweep(&linop, 0, 0, cor_smooth_alev, *res_finest[0], true);
              }
          }

          Real beta = beta_numer/beta_denom;
          cor_top.mult(beta);
       //}

       //skip_fillboundary = false;
       //for (int i = 0; i < num_top_async_smooth_sweeps; ++i) {
       //    linop.smooth(amrlev, mglev_top, cor_smooth_alev, res_top, skip_fillboundary);
       //}
       //skip_fillboundary = false;
       //for (int i = 0; i < num_top_async_smooth_sweeps; ++i) {
       //    linop.smooth(amrlev, mglev_top, *sol[amrlev], rhs[amrlev], skip_fillboundary);
       //}

       //skip_fillboundary = false;
       //for (int i = 0; i < num_top_async_smooth_sweeps; ++i) {
       //    linop.smooth(amrlev, mglev_top, cor_smooth_alev, *res_finest[0], skip_fillboundary);
       //}

       //skip_fillboundary = false;
       //for (int i = 0; i < num_top_async_smooth_sweeps; ++i) {
       //    //linop.smooth(amrlev, mglev_top, cor_smooth_alev, *res_finest[0], skip_fillboundary);
       //    asyncSmoothSweep(&linop, 0, 0, cor_smooth_alev, *res_finest[0], true);
       //}
       MultiFab::Add(cor_smooth_alev, cor_top, 0, 0, ncomp, nghost);
    }
}

void
MLAsyncMG::syncSmooth (int amrlev, int mglev, int num_smooth_sweeps, bool skip_fillboundary)
{
    BL_PROFILE("MLAsyncMG::syncSmooth()");

    MLCellLinOp *cell_linop = dynamic_cast<MLCellLinOp *>(&linop);
    MultiFab& cor_smooth_alev = *cor_smooth[amrlev];

    //printf("%d\n", mglev);
    //fflush(stdout);

    for (int i = 0; i < num_smooth_sweeps; ++i)
    {
        for (int redblack = 0; redblack < 2; ++redblack)
        {
            bool comm_complete;
            cell_linop->testInit(comm_complete);
            while (!comm_complete)
            {
                cell_linop->testSmooth(amrlev, mglev, *cor[amrlev][mglev], res[amrlev][mglev], comm_complete, redblack, skip_fillboundary);
		asyncSmoothSweep(&linop, 0, 0, cor_smooth_alev, *res_finest[0], true);
            }
            skip_fillboundary = false;
        }
    }
}

void
MLAsyncMG::asyncSmoothSweep (MLLinOp *async_linop,
                             int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs,
                             bool skip_fillboundary)
{
    BL_PROFILE("MLAsyncMG::asyncSmoothSweep()");

    MLCellLinOp *async_cell_linop = dynamic_cast<MLCellLinOp *>(async_linop);
    const int ncomp = async_cell_linop->getNComp();
    const int cross = async_cell_linop->isCrossStencil();
    const Geometry& m_geom = async_cell_linop->Geom(amrlev, mglev);

    if (asyncFillBoundary_count > 0)
    {
        sol.AsyncFillBoundary(0, ncomp, sol.nGrowVect(), m_geom.periodicity(), cross);
	asyncFillBoundary_count++;
    }
    for (int redblack = 0; redblack < 2; ++redblack)
    {
        //sol.FillBoundary(0, ncomp, m_geom.periodicity(), cross);

        async_cell_linop->applyBC(amrlev, mglev, sol, MLLinOp::BCMode::Homogeneous, MLLinOp::StateMode::Solution,
                                  nullptr, true);
//#ifdef AMREX_SOFT_PERF_COUNTERS
//        perf_counters.smooth(sol);
//#endif
        async_cell_linop->Fsmooth(amrlev, mglev, sol, rhs, redblack);

	sol.AsyncFillBoundary(0, ncomp, sol.nGrowVect(), m_geom.periodicity(), cross);
	asyncFillBoundary_count++;
    }
}

void
MLAsyncMG::testInit(bool& comm_complete)
{
    test_count = 0;
    comm_complete = false;
}

}
