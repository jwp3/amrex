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
        auto iter_start_time = amrex::second();
        bool converged = false;

        const int niters = do_fixed_number_of_iters ? do_fixed_number_of_iters : max_iters;

        

        for (int iter = 0; iter < niters; ++iter)
        {
            asyncOneIter(iter);

            converged = false;

            computeResidual(finest_amr_lev);

            if (is_nsolve) continue;

            Real fine_norminf = ResNormInf(finest_amr_lev);
            m_iter_fine_resnorm0.push_back(fine_norminf);
            composite_norminf = fine_norminf;
            if (verbose >= 2) {
                amrex::Print() << "MLAsyncMG: Iteration " << std::setw(3) << iter+1 << " Fine resid/"
                               << norm_name << " = " << fine_norminf/max_norm << "\n";
            }
            bool fine_converged = (fine_norminf <= res_target);

            if (namrlevs == 1 && fine_converged) {
                converged = true;
            } else if (fine_converged) {
                // finest level is converged, but we still need to test the coarse levels
                computeMLResidual(finest_amr_lev-1);
                Real crse_norminf = MLResNormInf(finest_amr_lev-1);
                if (verbose >= 2) {
                    amrex::Print() << "MLAsyncMG: Iteration " << std::setw(3) << iter+1
                                   << " Crse resid/" << norm_name << " = "
                                   << crse_norminf/max_norm << "\n";
                }
                converged = (crse_norminf <= res_target);
                composite_norminf = std::max(fine_norminf, crse_norminf);
            } else {
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

// in  : Residual (res) on the finest AMR level
// out : sol on all AMR levels
void MLAsyncMG::asyncOneIter (int iter)
{
    BL_PROFILE("MLMG::oneIter()");

    int ncomp = linop.getNComp();
    int nghost = 0;
    if (cf_strategy == CFStrategy::ghostnodes) nghost = linop.getNGrow();

    for (int alev = finest_amr_lev; alev > 0; --alev)
    {
        mgAddVcycle(alev, 0);

        MultiFab::Add(*sol[alev], *cor[alev][0], 0, 0, ncomp, nghost);

        // compute residual for the coarse AMR level
        computeResWithCrseSolFineCor(alev-1,alev);

        if (alev != finest_amr_lev) {
            std::swap(cor_hold[alev][0], cor[alev][0]); // save it for the up cycle
        }
    }

    // coarsest amr level
    {
        // enforce solvability if appropriate
        if (linop.isSingular(0) && linop.getEnforceSingularSolvable())
        {
            makeSolvable(0,0,res[0][0]);
        }
        
        mgAddVcycle(0, 0);

        MultiFab::Add(*sol[0], *cor[0][0], 0, 0, ncomp, 0);
    }

    for (int alev = 1; alev <= finest_amr_lev; ++alev)
    {
        // (Fine AMR correction) = I(Coarse AMR correction)
        interpCorrection(alev);

        MultiFab::Add(*sol[alev], *cor[alev][0], 0, 0, ncomp, nghost);

        if (alev != finest_amr_lev) {
            MultiFab::Add(*cor_hold[alev][0], *cor[alev][0], 0, 0, ncomp, nghost);
        }

        // Update fine AMR level correction
        computeResWithCrseCorFineCor(alev);

        mgAddVcycle(alev, 0);

        MultiFab::Add(*sol[alev], *cor[alev][0], 0, 0, ncomp, nghost);

        if (alev != finest_amr_lev) {
            MultiFab::Add(*cor[alev][0], *cor_hold[alev][0], 0, 0, ncomp, nghost);
        }
    }

    averageDownAndSync();
}

void
MLAsyncMG::prepareForAsyncSolve (const Vector<MultiFab*>& a_sol, const Vector<MultiFab const*>& a_rhs)
{
    prepareForSolve(a_sol, a_rhs);

    const int ncomp = linop.getNComp();
    IntVect ng_sol(1);
    if (linop.hasHiddenDimension()) ng_sol[linop.hiddenDirection()] = 0;

    IntVect ng = linop.isCellCentered() ? IntVect(0) : IntVect(1);
    if (cf_strategy != CFStrategy::ghostnodes) ng = ng_sol;

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

       cor_smooth_alev.setVal(0.0);
       int num_top_async_smooth_sweeps = 100;

       skip_fillboundary = true;
       for (int i = 0; i < num_top_async_smooth_sweeps; ++i)
       {
	   for (int redblack = 0; redblack < 2; ++redblack)
           {
               bool comm_complete = false;
               while (!comm_complete)
               {
                   cell_linop->smooth(amrlev, mglev_top, cor_smooth_alev, res_top, comm_complete, redblack, skip_fillboundary);
               }
	       skip_fillboundary = false;
           }
       }

       //skip_fillboundary = true;
       //for (int i = 0; i < num_top_async_smooth_sweeps; ++i) {
       //    linop.smooth(amrlev, mglev_top, cor_smooth_alev, res_top, skip_fillboundary);
       //    skip_fillboundary = false;
       //}

       MultiFab::Add(cor_top, cor_smooth_alev, 0, 0, ncomp, nghost);
       return;

       int sweep = 0;
       int local_converge = 0;
       cor_smooth_alev.AsyncSetup();
       while (1)
       {
           //linop.smooth(amrlev, mglev_top, cor_top, res_top, true);
           asyncSmoothSweep(&linop, amrlev, mglev_top, cor_smooth_alev, res_top, true);
           sweep++;
           if (sweep >= num_top_async_smooth_sweeps)
	   {
               local_converge = 1;
           }
           int converge_flag = cor_smooth_alev.AsyncCheckConverge(local_converge, num_top_async_smooth_sweeps);
           if (converge_flag == 1){
              break;
           }
       }
       const int cross = cell_linop->isCrossStencil();
       const Geometry& m_geom = cell_linop->Geom(amrlev, mglev_top);
       cor_smooth_alev.AsyncCleanup(0, ncomp, cor_smooth_alev.nGrowVect(), m_geom.periodicity(), cross);

       MultiFab::Add(cor_top, cor_smooth_alev, 0, 0, ncomp, nghost);
       MPI_Comm comm = ParallelContext::CommunicatorSub();
       int rank = ParallelDescriptor::MyProc(comm);
     //  printf("%d %d\n", rank, sweep);
     //  fflush(stdout);

       return;

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
                else {
                   cor[amrlev][mglev]->setVal(0.0);
                   skip_fillboundary = true;
                }
                for (int i = 0; i < 1; ++i)
                {
                    linop.smooth(amrlev, mglev, *cor[amrlev][mglev], res[amrlev][mglev],
                                 skip_fillboundary);
                    skip_fillboundary = false;
                }
                computeResOfCorrection(amrlev, mglev);
                linop.restriction(amrlev, mglev+1, res[amrlev][mglev+1], rescor[amrlev][mglev]);
             }

             bottomSolve();

             for (int mglev = mglev_bottom-1; mglev >= mglev_top+1; --mglev)
             {
                addInterpCorrection(amrlev, mglev);
                skip_fillboundary = false;
                for (int i = 0; i < 1; ++i)
                {
                    linop.smooth(amrlev, mglev, *cor[amrlev][mglev], res[amrlev][mglev],
                                 skip_fillboundary);
                }
             }
          }

          cor[amrlev][mglev_top]->setVal(0.0);
          addInterpCorrection(amrlev, mglev_top);
       }

       //if (0) { 
          cor_smooth_alev.setVal(0.0);
          skip_fillboundary = true;
          for (int i = 0; i < num_top_async_smooth_sweeps; ++i) {
              linop.smooth(amrlev, mglev_top, cor_smooth_alev, res_top, skip_fillboundary);
              skip_fillboundary = false;
          }
 
          MultiFab& Acor_alev = *Acor[amrlev];
          MultiFab& rescor_top = rescor[amrlev][mglev_top];

          linop.correctionResidual(amrlev, mglev_top, rescor_top, cor_smooth_alev, res_top, BCMode::Homogeneous);

          Real beta_numer = linop.xdoty(amrlev, mglev_top, cor_top, rescor_top, false);
          linop.apply(amrlev, mglev_top, Acor_alev, cor_top, MLLinOp::BCMode::Homogeneous, MLLinOp::StateMode::Correction);
          Real beta_denom = linop.xdoty(amrlev, mglev_top, cor_top, Acor_alev, false);

          Real beta = beta_numer/beta_denom;
          cor_top.mult(beta);
       //}

       //cor_smooth_alev.setVal(0.0);
       //skip_fillboundary = true;
       for (int i = 0; i < num_top_async_smooth_sweeps; ++i) {
           linop.smooth(amrlev, mglev_top, cor_smooth_alev, res_top, skip_fillboundary);
           //skip_fillboundary = false;
       }

       MultiFab::Add(cor_top, cor_smooth_alev, 0, 0, ncomp, nghost);
    }
}


void
MLAsyncMG::asyncSmoothSweep (MLLinOp *async_linop,
                             int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs,
                             bool skip_fillboundary)
{
    BL_PROFILE("asyncSmoothSweep()");

    MLCellLinOp *async_cell_linop = dynamic_cast<MLCellLinOp *>(async_linop);
    const int ncomp = async_cell_linop->getNComp();
    const int cross = async_cell_linop->isCrossStencil();
    const Geometry& m_geom = async_cell_linop->Geom(amrlev, mglev);

    //sol.AsyncFillBoundary(0, ncomp, sol.nGrowVect(), m_geom.periodicity(), cross);
    for (int redblack = 0; redblack < 2; ++redblack)
    {
        //sol.FillBoundary(0, ncomp, m_geom.periodicity(), cross);

        async_cell_linop->applyBC(amrlev, mglev, sol, MLLinOp::BCMode::Homogeneous, MLLinOp::StateMode::Solution,
                                  nullptr, true);
//#ifdef AMREX_SOFT_PERF_COUNTERS
//        perf_counters.smooth(sol);
//#endif
        async_cell_linop->Fsmooth(amrlev, mglev, sol, rhs, redblack);
    }
    sol.AsyncFillBoundary(0, ncomp, sol.nGrowVect(), m_geom.periodicity(), cross);
}

}
