#ifndef AMREX_ML_ASYNC_MG_H_
#define AMREX_ML_ASYNC_MG_H_
#include <AMReX_Config.H>

#include <AMReX_MLMG.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MLCGSolver.H>

#if defined(AMREX_USE_HYPRE) && (AMREX_SPACEDIM > 1)
#include <AMReX_Hypre.H>
#include <AMReX_HypreNodeLap.H>
#endif

namespace amrex {

class MLAsyncMG
    : public MLMG
{
public:
    friend class MLCGSolver;

    using BCMode = MLLinOp::BCMode;
    using Location = MLLinOp::Location;

    using BottomSolver = amrex::BottomSolver;

    MLAsyncMG (MLLinOp& a_lp);
    ~MLAsyncMG ();

    Real asyncSolve (const Vector<MultiFab*>& a_sol, const Vector<MultiFab const*>& a_rhs,
                     Real a_tol_rel, Real a_tol_abs, const char* checkpoint_file = nullptr);
    void asyncOneIter (int iter);
    void prepareForAsyncSolve (const Vector<MultiFab*>& a_sol, const Vector<MultiFab const*>& a_rhs);
    void mgAddVcycle (int amrlev, int mglev_top);
    void asyncSmoothSweep (MLLinOp *async_linop,
                           int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs,
                           bool skip_fillboundary);

    Vector<std::unique_ptr<MultiFab>> cor_smooth;
    Vector<std::unique_ptr<MultiFab>> Acor;
};

}

#endif
