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
    void testComputeResidual (int alev);
    void testComputeResOfCorrection (int amrlev, int mglev);
    void syncSmooth (int amrlev, int mglev, int num_smooth_sweeps, bool skip_fillboundary = false);
    Real testResNormInf (int alev, bool& comm_complete, bool local = false);
    void testInit(bool& comm_complete);

    Vector<std::unique_ptr<MultiFab>> cor_finest;
    Vector<std::unique_ptr<MultiFab>> res_finest;
    Vector<std::unique_ptr<MultiFab>> cor_smooth;
    Vector<std::unique_ptr<MultiFab>> Acor;

    int test_count = 0;
    MPI_Request reduce_req;
    MPI_Status reduce_stat;
    Real *reduce_recv;
    Real *reduce_send;

    int asyncFillBoundary_count = 0;
};

}

#endif
