
#include <AMReX_FBI.H>
#include <AMReX_PCI.H>


//TODO: handle multple amr levels.

template <class FAB>
template <class F, typename std::enable_if<IsBaseFab<F>::value,int>::type Z>
void
FabArray<FAB>::FBEP_nowait (int scomp, int ncomp, const IntVect& nghost,
                            const Periodicity& period, bool cross,
                            bool enforce_periodicity_only)
{
    AMREX_ASSERT_WITH_MESSAGE(!fbd, "FillBoundary_nowait() called when comm operation already in progress.");

    ///* begin JORDI section */
    //if (fbd) delete fbd;
    //fbd = new FBData<FAB>();
    ///* end JORDI section */

    bool work_to_do;
    if (enforce_periodicity_only) {
        work_to_do = period.isAnyPeriodic();
    } else {
        work_to_do = nghost.max() > 0;
    }
    if (!work_to_do) { return; }

    const FB& TheFB = getFB(nghost, period, cross, enforce_periodicity_only);

    if (ParallelContext::NProcsSub() == 1)
    {
        //
        // There can only be local work to do.
        //
        int N_locs = (*TheFB.m_LocTags).size();
        if (N_locs == 0) return;
#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10))
            if (Gpu::inGraphRegion())
            {
                FB_local_copy_cuda_graph_1(TheFB, scomp, ncomp);
            }
            else
#endif
            {
                FB_local_copy_gpu(TheFB, scomp, ncomp);
            }
        }
        else
#endif
        {
            FB_local_copy_cpu(TheFB, scomp, ncomp);
        }

        return;
    }

#ifdef BL_USE_MPI

    //
    // Do this before prematurely exiting if running in parallel.
    // Otherwise sequence numbers will not match across MPI processes.
    //
    int SeqNum = ParallelDescriptor::SeqNum();

    const int N_locs = TheFB.m_LocTags->size();
    const int N_rcvs = TheFB.m_RcvTags->size();
    const int N_snds = TheFB.m_SndTags->size();

    if (N_locs == 0 && N_rcvs == 0 && N_snds == 0) {
        // No work to do.
        return;
    }

    /* begin JORDI section */
    //fbd = std::make_unique<FBData<FAB>>();
    fbd = new FBData<FAB>();
    /* end JORDI section */

    fbd->fb    = &TheFB;
    fbd->scomp = scomp;
    fbd->ncomp = ncomp;
    fbd->nghost = nghost;
    fbd->period = period;
    fbd->cross = cross;
    fbd->epo   = enforce_periodicity_only;
    fbd->tag   = SeqNum;

    //
    // Post rcvs. Allocate one chunk of space to hold'm all.
    //

    if (N_rcvs > 0) {
        PostRcvs(*TheFB.m_RcvTags, fbd->the_recv_data,
                 fbd->recv_data, fbd->recv_size, fbd->recv_from, fbd->recv_reqs,
                 ncomp, SeqNum);
        fbd->recv_stat.resize(N_rcvs);
    }

    //
    // Post send's
    //
    char*&                          the_send_data = fbd->the_send_data;
    Vector<char*> &                     send_data = fbd->send_data;
    Vector<std::size_t>                 send_size;
    Vector<int>                         send_rank;
    Vector<MPI_Request>&                send_reqs = fbd->send_reqs;
    Vector<const CopyComTagsContainer*> send_cctc;

    if (N_snds > 0)
    {
        PrepareSendBuffers(*TheFB.m_SndTags, the_send_data, send_data, send_size, send_rank,
                           send_reqs, send_cctc, ncomp);

#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10))
            if (Gpu::inGraphRegion()) {
                FB_pack_send_buffer_cuda_graph(TheFB, scomp, ncomp, send_data, send_size, send_cctc);
            }
            else
#endif
            {
                pack_send_buffer_gpu(*this, scomp, ncomp, send_data, send_size, send_cctc);
            }
        }
        else
#endif
        {
            pack_send_buffer_cpu(*this, scomp, ncomp, send_data, send_size, send_cctc);
        }

        AMREX_ASSERT(send_reqs.size() == N_snds);
        PostSnds(send_data, send_size, send_rank, send_reqs, SeqNum);
    }

    /* begin JORDI section */
    //FillBoundary_test();
    /* end JORDI section */

    //
    // Do the local work.  Hope for a bit of communication/computation overlap.
    //
    if (N_locs > 0)
    {
#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10) )
            if (Gpu::inGraphRegion()) {
                FB_local_copy_cuda_graph_n(TheFB, scomp, ncomp);
            }
            else
#endif
            {
                FB_local_copy_gpu(TheFB, scomp, ncomp);
            }
        }
        else
#endif
        {
            FB_local_copy_cpu(TheFB, scomp, ncomp);
        }

	/* begin JORDI section */
        //FillBoundary_test();
	/* end JORDI section */
    }

#endif /*BL_USE_MPI*/
}

template <class FAB>
template <class F, typename std::enable_if<IsBaseFab<F>::value,int>::type Z>
void
FabArray<FAB>::FillBoundary_finish ()
{
#ifdef AMREX_USE_MPI

    BL_PROFILE("FillBoundary_finish()");

    if (!fbd) { n_filled = IntVect::TheZeroVector(); return; }

    const FB* TheFB = fbd->fb;
    const int N_rcvs = TheFB->m_RcvTags->size();
    if (N_rcvs > 0)
    {
        Vector<const CopyComTagsContainer*> recv_cctc(N_rcvs,nullptr);
        for (int k = 0; k < N_rcvs; k++)
        {
            if (fbd->recv_size[k] > 0)
            {
                auto const& cctc = TheFB->m_RcvTags->at(fbd->recv_from[k]);
                recv_cctc[k] = &cctc;
            }
        }

        int actual_n_rcvs = N_rcvs - std::count(fbd->recv_data.begin(), fbd->recv_data.end(), nullptr);

        if (actual_n_rcvs > 0) {
            ParallelDescriptor::Waitall(fbd->recv_reqs, fbd->recv_stat);
#ifdef AMREX_DEBUG
            if (!CheckRcvStats(fbd->recv_stat, fbd->recv_size, fbd->tag))
            {
                amrex::Abort("FillBoundary_finish failed with wrong message size");
            }
#endif
        }

        bool is_thread_safe = TheFB->m_threadsafe_rcv;

#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10) )
            if (Gpu::inGraphRegion())
            {
                FB_unpack_recv_buffer_cuda_graph(*TheFB, fbd->scomp, fbd->ncomp,
                                                 fbd->recv_data, fbd->recv_size,
                                                 recv_cctc, is_thread_safe);
            }
            else
#endif
            {
                unpack_recv_buffer_gpu(*this, fbd->scomp, fbd->ncomp, fbd->recv_data, fbd->recv_size,
                                       recv_cctc, FabArrayBase::COPY, is_thread_safe);
            }
        }
        else
#endif
        {
            unpack_recv_buffer_cpu(*this, fbd->scomp, fbd->ncomp, fbd->recv_data, fbd->recv_size,
                                   recv_cctc, FabArrayBase::COPY, is_thread_safe);
        }

        if (fbd->the_recv_data)
        {
            amrex::The_FA_Arena()->free(fbd->the_recv_data);
            fbd->the_recv_data = nullptr;
        }
    }

    const int N_snds = TheFB->m_SndTags->size();
    if (N_snds > 0) {
        Vector<MPI_Status> stats(fbd->send_reqs.size());
        ParallelDescriptor::Waitall(fbd->send_reqs, stats);
        amrex::The_FA_Arena()->free(fbd->the_send_data);
        fbd->the_send_data = nullptr;
    }


    /* begin JORDI section */
    delete fbd;
    fbd = nullptr;
    //fbd.reset();
    /* end JORDI section */
#endif
}

template <class FAB>
void
FabArray<FAB>::ParallelCopy (const FabArray<FAB>& src,
                             int                  scomp,
                             int                  dcomp,
                             int                  ncomp,
                             const IntVect&       snghost,
                             const IntVect&       dnghost,
                             const Periodicity&   period,
                             CpOp                 op,
                             const FabArrayBase::CPC * a_cpc)
{
    BL_PROFILE("FabArray::ParallelCopy()");

    ParallelCopy_nowait(src, scomp, dcomp, ncomp, snghost, dnghost, period, op, a_cpc);
    ParallelCopy_finish();
}

template <class FAB>
void
FabArray<FAB>::ParallelCopy_nowait (const FabArray<FAB>& src,
                                    int                  scomp,
                                    int                  dcomp,
                                    int                  ncomp,
                                    const IntVect&       snghost,
                                    const IntVect&       dnghost,
                                    const Periodicity&   period,
                                    CpOp                 op,
                                    const FabArrayBase::CPC * a_cpc)
{
    BL_PROFILE("FabArray::ParallelCopy_nowait()");

    AMREX_ASSERT_WITH_MESSAGE(!pcd, "ParallelCopy_nowait() called when comm operation already in progress.");

    if (size() == 0 || src.size() == 0) {
        return;
    }

    BL_ASSERT(op == FabArrayBase::COPY || op == FabArrayBase::ADD);
    BL_ASSERT(boxArray().ixType() == src.boxArray().ixType());
    BL_ASSERT(src.nGrowVect().allGE(snghost));
    BL_ASSERT(    nGrowVect().allGE(dnghost));

    n_filled = dnghost;

    if ((src.boxArray().ixType().cellCentered() || op == FabArrayBase::COPY) &&
        (boxarray == src.boxarray && distributionMap == src.distributionMap) &&
        snghost == IntVect::TheZeroVector() &&
        dnghost == IntVect::TheZeroVector() &&
        !period.isAnyPeriodic())
    {
        //
        // Short-circuit full intersection code if we're doing copy()s or if
        // we're doing plus()s on cell-centered data.  Don't do plus()s on
        // non-cell-centered data this simplistic way.
        //
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter fai(*this,TilingIfNotGPU()); fai.isValid(); ++fai)
        {
            const Box& bx = fai.tilebox();

            // avoid self copy or plus
            if (this != &src) {
                auto const sfab = src.array(fai);
                auto       dfab = this->array(fai);
                if (op == FabArrayBase::COPY) {
                    AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
                    {
                        dfab(i,j,k,dcomp+n) = sfab(i,j,k,scomp+n);
                    });
                } else {
                    AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
                    {
                        dfab(i,j,k,dcomp+n) += sfab(i,j,k,scomp+n);
                    });
                }
            }
        }

        return;
    }

    const CPC& thecpc = (a_cpc) ? *a_cpc : getCPC(dnghost, src, snghost, period);

    if (ParallelContext::NProcsSub() == 1)
    {
        //
        // There can only be local work to do.
        //

        int N_locs = (*thecpc.m_LocTags).size();
        if (N_locs == 0) { return; }
#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
            PC_local_gpu(thecpc, src, scomp, dcomp, ncomp, op);
        }
        else
#endif
        {
            PC_local_cpu(thecpc, src, scomp, dcomp, ncomp, op);
        }

        return;
    }

#ifdef BL_USE_MPI

    //
    // Do this before prematurely exiting if running in parallel.
    // Otherwise sequence numbers will not match across MPI processes.
    //
    int tag = ParallelDescriptor::SeqNum();

    const int N_snds = thecpc.m_SndTags->size();
    const int N_rcvs = thecpc.m_RcvTags->size();
    const int N_locs = thecpc.m_LocTags->size();

    if (N_locs == 0 && N_rcvs == 0 && N_snds == 0) {
        //
        // No work to do.
        //

        return;
    }

    //
    // Send/Recv at most MaxComp components at a time to cut down memory usage.
    //
    int NCompLeft = ncomp;
    int SC = scomp, DC = dcomp, NC;

    for (int ipass = 0; ipass < ncomp; )
    {
        pcd = std::make_unique<PCData<FAB>>();
        pcd->cpc = &thecpc;
        pcd->src = &src;
        pcd->snghost = snghost;
        pcd->dnghost = dnghost;
        pcd->period = period;
        pcd->op = op;
        pcd->tag = tag;

        NC = std::min(NCompLeft,FabArrayBase::MaxComp);
        const bool last_iter = (NCompLeft == NC);

        pcd->SC = SC;
        pcd->DC = DC;
        pcd->NC = NC;

        //
        // Post rcvs. Allocate one chunk of space to hold'm all.
        //
        pcd->the_recv_data = nullptr;

        pcd->actual_n_rcvs = 0;
        if (N_rcvs > 0) {
            PostRcvs(*thecpc.m_RcvTags, pcd->the_recv_data,
                     pcd->recv_data, pcd->recv_size, pcd->recv_from, pcd->recv_reqs, NC, pcd->tag);
            pcd->actual_n_rcvs = N_rcvs - std::count(pcd->recv_size.begin(), pcd->recv_size.end(), 0);
        }

        //
        // Post send's
        //
        Vector<char*>                       send_data;
        Vector<std::size_t>                 send_size;
        Vector<int>                         send_rank;
        Vector<const CopyComTagsContainer*> send_cctc;

        if (N_snds > 0)
        {
            src.PrepareSendBuffers(*thecpc.m_SndTags, pcd->the_send_data, send_data, send_size,
                                   send_rank, pcd->send_reqs, send_cctc, NC);

#ifdef AMREX_USE_GPU
            if (Gpu::inLaunchRegion())
            {
                pack_send_buffer_gpu(src, SC, NC, send_data, send_size, send_cctc);
            }
            else
#endif
            {
                pack_send_buffer_cpu(src, SC, NC, send_data, send_size, send_cctc);
            }

            AMREX_ASSERT(pcd->send_reqs.size() == N_snds);
            FabArray<FAB>::PostSnds(send_data, send_size, send_rank, pcd->send_reqs, pcd->tag);
        }

        //
        // Do the local work.  Hope for a bit of communication/computation overlap.
        //
        if (N_locs > 0)
        {
#ifdef AMREX_USE_GPU
            if (Gpu::inLaunchRegion())
            {
                PC_local_gpu(thecpc, src, SC, DC, NC, op);
            }
            else
#endif
            {
                PC_local_cpu(thecpc, src, SC, DC, NC, op);
            }
        }

        if (!last_iter)
        {
            ParallelCopy_finish();

            SC += NC;
            DC += NC;
        }

        ipass     += NC;
        NCompLeft -= NC;
    }

#endif /*BL_USE_MPI*/
}

template <class FAB>
void
FabArray<FAB>::ParallelCopy_finish ()
{

#ifdef BL_USE_MPI

    BL_PROFILE("FabArray::ParallelCopy_finish()");

    if (!pcd) { return; }

    const CPC* thecpc = pcd->cpc;

    const int N_snds = thecpc->m_SndTags->size();
    const int N_rcvs = thecpc->m_RcvTags->size();

    if (N_rcvs > 0)
    {
        Vector<const CopyComTagsContainer*> recv_cctc(N_rcvs,nullptr);
        for (int k = 0; k < N_rcvs; ++k)
        {
            if (pcd->recv_size[k] > 0)
            {
                auto const& cctc = thecpc->m_RcvTags->at(pcd->recv_from[k]);
                recv_cctc[k] = &cctc;
            }
        }

        if (pcd->actual_n_rcvs > 0) {
            Vector<MPI_Status> stats(N_rcvs);
            ParallelDescriptor::Waitall(pcd->recv_reqs, stats);
#ifdef AMREX_DEBUG
            if (!CheckRcvStats(stats, pcd->recv_size, pcd->tag))
            {
                amrex::Abort("ParallelCopy failed with wrong message size");
            }
#endif
        }

        bool is_thread_safe = thecpc->m_threadsafe_rcv;

#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
            unpack_recv_buffer_gpu(*this, pcd->DC, pcd->NC, pcd->recv_data, pcd->recv_size,
                                   recv_cctc, pcd->op, is_thread_safe);
        }
        else
#endif
        {
            unpack_recv_buffer_cpu(*this, pcd->DC, pcd->NC, pcd->recv_data, pcd->recv_size,
                                   recv_cctc, pcd->op, is_thread_safe);
        }

        if (pcd->the_recv_data)
        {
            amrex::The_FA_Arena()->free(pcd->the_recv_data);
            pcd->the_recv_data = nullptr;
        }
    }

    if (N_snds > 0) {
        if (! thecpc->m_SndTags->empty()) {
            Vector<MPI_Status> stats(pcd->send_reqs.size());
            ParallelDescriptor::Waitall(pcd->send_reqs, stats);
        }
        amrex::The_FA_Arena()->free(pcd->the_send_data);
        pcd->the_send_data = nullptr;
    }

    pcd.reset();

#endif /*BL_USE_MPI*/

}

template <class FAB>
void
FabArray<FAB>::copyTo (FAB&       dest,
                       const Box& subbox,
                       int        scomp,
                       int        dcomp,
                       int        ncomp,
                       int        nghost) const
{
    BL_PROFILE("FabArray::copy(fab)");

    BL_ASSERT(dcomp + ncomp <= dest.nComp());
    BL_ASSERT(nghost <= nGrow());

    if (ParallelContext::NProcsSub() == 1)
    {
        for (int j = 0, N = size(); j < N; ++j)
        {
            const Box& bx = amrex::grow(boxarray[j],nghost);
            const Box& destbox = bx & subbox;
            if (destbox.ok())
            {
                dest.template copy<RunOn::Host>(get(j),destbox,scomp,destbox,dcomp,ncomp);
            }
        }

        return;
    }

    //
    //  Note that subbox must be identical on each process!!
    //
#ifdef AMREX_DEBUG
    {
        BoxCommHelper bch(subbox);
        ParallelDescriptor::Bcast(bch.data(), bch.size(), 0, ParallelContext::CommunicatorSub());
        const Box& bx0 = bch.make_box();
        BL_ASSERT(subbox == bx0);
    }
#endif

    FAB ovlp;

    std::vector< std::pair<int,Box> > isects;
    boxarray.intersections(subbox, isects, false, nghost);

    for (int j = 0, M = isects.size(); j < M; ++j)
    {
        const int  k  = isects[j].first;
        const Box& bx = isects[j].second;

        ovlp.resize(bx,ncomp);

        if (ParallelDescriptor::MyProc() == distributionMap[k])
        {
            ovlp.template copy<RunOn::Host>(get(k),bx,scomp,bx,0,ncomp);
        }

        const int N = bx.numPts()*ncomp;

        ParallelDescriptor::Bcast(ovlp.dataPtr(),N,
                                  ParallelContext::global_to_local_rank(distributionMap[k]),
                                  ParallelContext::CommunicatorSub());

        dest.template copy<RunOn::Host>(ovlp,bx,0,bx,dcomp,ncomp);
    }
}


#ifdef BL_USE_MPI
template <class FAB>
AMREX_NODISCARD TheFaArenaPointer
FabArray<FAB>::PrepareSendBuffers (const MapOfCopyComTagContainers&     SndTags,
                                   Vector<char*>&                       send_data,
                                   Vector<std::size_t>&                 send_size,
                                   Vector<int>&                         send_rank,
                                   Vector<MPI_Request>&                 send_reqs,
                                   Vector<const CopyComTagsContainer*>& send_cctc,
                                   int                                  ncomp) const
{
    char* pointer = nullptr;
    PrepareSendBuffers(SndTags, pointer, send_data, send_size, send_rank, send_reqs, send_cctc, ncomp);
    return TheFaArenaPointer(pointer);
}

template <class FAB>
void
FabArray<FAB>::PrepareSendBuffers (const MapOfCopyComTagContainers&     SndTags,
                                   char*&                               the_send_data,
                                   Vector<char*>&                       send_data,
                                   Vector<std::size_t>&                 send_size,
                                   Vector<int>&                         send_rank,
                                   Vector<MPI_Request>&                 send_reqs,
                                   Vector<const CopyComTagsContainer*>& send_cctc,
                                   int                                  ncomp) const
{
    send_data.clear();
    send_size.clear();
    send_rank.clear();
    send_reqs.clear();
    send_cctc.clear();
    const int N_snds = SndTags.size();
    if (N_snds == 0) return;
    send_data.reserve(N_snds);
    send_size.reserve(N_snds);
    send_rank.reserve(N_snds);
    send_reqs.reserve(N_snds);
    send_cctc.reserve(N_snds);

    Vector<std::size_t> offset; offset.reserve(N_snds);
    std::size_t total_volume = 0;
    for (auto const& kv : SndTags)
    {
        Vector<int> iss;
        auto const& cctc = kv.second;

        std::size_t nbytes = 0;
        for (auto const& cct : kv.second)
        {
            nbytes += (*this)[cct.srcIndex].nBytes(cct.sbox,ncomp);
        }

        std::size_t acd = ParallelDescriptor::alignof_comm_data(nbytes);
        nbytes = amrex::aligned_size(acd, nbytes); // so that bytes are aligned

        // Also need to align the offset properly
        total_volume = amrex::aligned_size(std::max(alignof(typename FAB::value_type),
                                                    acd),
                                           total_volume);

        offset.push_back(total_volume);
        total_volume += nbytes;

        send_data.push_back(nullptr);
        send_size.push_back(nbytes);
        send_rank.push_back(kv.first);
        send_reqs.push_back(MPI_REQUEST_NULL);
        send_cctc.push_back(&cctc);
    }

    if (total_volume > 0)
    {
        the_send_data = static_cast<char*>(amrex::The_FA_Arena()->alloc(total_volume));
        for (int i = 0, N = send_size.size(); i < N; ++i) {
            send_data[i] = the_send_data + offset[i];
        }
    } else {
        the_send_data = nullptr;
    }
}

template <class FAB>
void
FabArray<FAB>::PostSnds (Vector<char*> const&       send_data,
                         Vector<std::size_t> const& send_size,
                         Vector<int> const&         send_rank,
                         Vector<MPI_Request>&       send_reqs,
                         int                        SeqNum)
{
    MPI_Comm comm = ParallelContext::CommunicatorSub();

    const int N_snds = send_reqs.size();
    for (int j = 0; j < N_snds; ++j)
    {
        if (send_size[j] > 0) {
            const int rank = ParallelContext::global_to_local_rank(send_rank[j]);
            send_reqs[j] = ParallelDescriptor::Asend
                (send_data[j], send_size[j], rank, SeqNum, comm).req();
        }
    }
}

template <class FAB>
TheFaArenaPointer FabArray<FAB>::PostRcvs (const MapOfCopyComTagContainers&       RcvTags,
                   Vector<char*>&                         recv_data,
                   Vector<std::size_t>&                   recv_size,
                   Vector<int>&                           recv_from,
                   Vector<MPI_Request>&                   recv_reqs,
                   int                                    ncomp,
                   int                                    SeqNum) const
{
    char* pointer = nullptr;
    PostRcvs(RcvTags, pointer, recv_data, recv_size, recv_from, recv_reqs, ncomp, SeqNum);
    return TheFaArenaPointer(pointer);
}

template <class FAB>
void
FabArray<FAB>::PostRcvs (const MapOfCopyComTagContainers&  RcvTags,
                         char*&                            the_recv_data,
                         Vector<char*>&                    recv_data,
                         Vector<std::size_t>&              recv_size,
                         Vector<int>&                      recv_from,
                         Vector<MPI_Request>&              recv_reqs,
                         int                               ncomp,
                         int                               SeqNum) const
{
    recv_data.clear();
    recv_size.clear();
    recv_from.clear();
    recv_reqs.clear();

    Vector<std::size_t> offset;
    std::size_t TotalRcvsVolume = 0;
    for (const auto& kv : RcvTags) // loop over senders
    {
        std::size_t nbytes = 0;
        for (auto const& cct : kv.second)
        {
            nbytes += (*this)[cct.dstIndex].nBytes(cct.dbox,ncomp);
        }

        std::size_t acd = ParallelDescriptor::alignof_comm_data(nbytes);
        nbytes = amrex::aligned_size(acd, nbytes);  // so that nbytes are aligned

        // Also need to align the offset properly
        TotalRcvsVolume = amrex::aligned_size(std::max(alignof(typename FAB::value_type),acd),
                                              TotalRcvsVolume);

        offset.push_back(TotalRcvsVolume);
        TotalRcvsVolume += nbytes;

        recv_data.push_back(nullptr);
        recv_size.push_back(nbytes);
        recv_from.push_back(kv.first);
        recv_reqs.push_back(MPI_REQUEST_NULL);
    }

    const int nrecv = recv_from.size();

    MPI_Comm comm = ParallelContext::CommunicatorSub();

    if (TotalRcvsVolume == 0)
    {
        the_recv_data = nullptr;
    }
    else
    {
        the_recv_data = static_cast<char*>(amrex::The_FA_Arena()->alloc(TotalRcvsVolume));

        for (int i = 0; i < nrecv; ++i)
        {
            recv_data[i] = the_recv_data + offset[i];
            if (recv_size[i] > 0)
            {
                const int rank = ParallelContext::global_to_local_rank(recv_from[i]);
                recv_reqs[i] = ParallelDescriptor::Arecv
                    (recv_data[i], recv_size[i], rank, SeqNum, comm).req();
            }
        }
    }
}
#endif

template <class FAB>
void
FabArray<FAB>::Redistribute (const FabArray<FAB>& src,
                             int                  scomp,
                             int                  dcomp,
                             int                  ncomp,
                             const IntVect&       nghost)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(boxArray() == src.boxArray(),
                                     "FabArray::Redistribute: must have the same BoxArray");

    if (ParallelContext::NProcsSub() == 1)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter fai(*this,true); fai.isValid(); ++fai)
        {
            const Box& bx = fai.growntilebox(nghost);
            auto const sfab = src.array(fai);
            auto       dfab = this->array(fai);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
            {
                dfab(i,j,k,n+dcomp) = sfab(i,j,k,n+scomp);
            });
        }

        return;
    }

#ifdef BL_USE_MPI

    FabArrayBase::CPC cpc(boxArray(), nghost, DistributionMap(), src.DistributionMap());

    ParallelCopy(src, scomp, dcomp, ncomp, nghost, nghost, Periodicity::NonPeriodic(),
                 FabArrayBase::COPY, &cpc);

#endif
}

template <class FAB>
void
FabArray<FAB>::FillBoundary_test ()
{
#if defined(AMREX_USE_MPI) && !defined(AMREX_DEBUG)
    // We only test if no DEBUG because in DEBUG we check the status later.
    // If Test is done here, the status check will fail.
    int flag;
    ParallelDescriptor::Test(fbd->recv_reqs, flag, fbd->recv_stat);
#endif
}

namespace detail {
template <class TagT>
void fbv_copy (Vector<TagT> const& tags)
{
    const int N = tags.size();
    if (N == 0) return;
#ifdef AMREX_USE_GPU
    if (Gpu::inLaunchRegion()) {
        ParallelFor(tags, 1,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int, TagT const& tag) noexcept
        {
            const int ncomp = tag.dfab.nComp();
            for (int n = 0; n < ncomp; ++n) {
                tag.dfab(i,j,k,n) = tag.sfab(i+tag.offset.x,j+tag.offset.y,k+tag.offset.z,n);
            }
        });
    } else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (int itag = 0; itag < N; ++itag) {
            auto const& tag = tags[itag];
            const int ncomp = tag.dfab.nComp();
            AMREX_LOOP_4D(tag.dbox, ncomp, i, j, k, n,
            {
                tag.dfab(i,j,k,n) = tag.sfab(i+tag.offset.x,j+tag.offset.y,k+tag.offset.z,n);
            });
        }
    }
}
}

template <class MF>
std::enable_if_t<IsFabArray<MF>::value>
FillBoundary (Vector<MF*> const& mf, Vector<int> const& scomp,
              Vector<int> const& ncomp, Vector<IntVect> const& nghost,
              Vector<Periodicity> const& period, Vector<int> const& cross = {})
{
    BL_PROFILE("FillBoundary(Vector)");
#if 1
    const int N = mf.size();
    for (int i = 0; i < N; ++i) {
        mf[i]->FillBoundary_nowait(scomp[i], ncomp[i], nghost[i], period[i],
                                   cross.empty() ? 0 : cross[i]);
    }
    for (int i = 0; i < N; ++i) {
        mf[i]->FillBoundary_finish();
    }

#else
    using FAB = typename MF::FABType::value_type;
    using T   = typename FAB::value_type;

    const int nmfs = mf.size();
    Vector<FabArrayBase::CommMetaData const*> cmds;
    int N_locs = 0;
    int N_rcvs = 0;
    int N_snds = 0;
    for (int imf = 0; imf < nmfs; ++imf) {
        if (nghost[imf].max() > 0) {
            auto const& TheFB = mf[imf]->getFB(nghost[imf], period[imf],
                                               cross.empty() ? 0 : cross[imf]);
            // The FB is cached.  Therefore it's safe take its address for later use.
            cmds.push_back(static_cast<FabArrayBase::CommMetaData const*>(&TheFB));
            N_locs += TheFB.m_LocTags->size();
            N_rcvs += TheFB.m_RcvTags->size();
            N_snds += TheFB.m_SndTags->size();
        } else {
            cmds.push_back(nullptr);
        }
    }

    using TagT = Array4CopyTag<T>;
    Vector<TagT> local_tags;
    local_tags.reserve(N_locs);
    static_assert(amrex::IsStoreAtomic<T>::value, "FillBoundary(Vector): storing T is not atomic");
    for (int imf = 0; imf < nmfs; ++imf) {
        if (cmds[imf]) {
            auto const& tags = *(cmds[imf]->m_LocTags);
            for (auto const& tag : tags) {
                local_tags.push_back({(*mf[imf])[tag.dstIndex].array      (scomp[imf],ncomp[imf]),
                                      (*mf[imf])[tag.srcIndex].const_array(scomp[imf],ncomp[imf]),
                                      tag.dbox,
                                      (tag.sbox.smallEnd()-tag.dbox.smallEnd()).dim3()});
            }
        }
    }

    if (ParallelContext::NProcsSub() == 1) {
        detail::fbv_copy(local_tags);
        return;
    }

#ifdef AMREX_USE_MPI
    //
    // Do this before prematurely exiting if running in parallel.
    // Otherwise sequence numbers will not match across MPI processes.
    //
    int SeqNum = ParallelDescriptor::SeqNum();
    MPI_Comm comm = ParallelContext::CommunicatorSub();

    if (N_locs == 0 && N_rcvs == 0 && N_snds == 0) return; // No work to do

    char* the_recv_data = nullptr;
    Vector<int> recv_from;
    Vector<std::size_t> recv_size;
    Vector<MPI_Request> recv_reqs;
    Vector<MPI_Status> recv_stat;
    Vector<TagT> recv_tags;

    if (N_rcvs > 0) {

        for (int imf = 0; imf < nmfs; ++imf) {
            if (cmds[imf]) {
                auto const& tags = *(cmds[imf]->m_RcvTags);
                for (const auto& kv : tags) {
                    recv_from.push_back(kv.first);
                }
            }
        }
        amrex::RemoveDuplicates(recv_from);
        const int nrecv = recv_from.size();

        recv_reqs.resize(nrecv, MPI_REQUEST_NULL);
        recv_stat.resize(nrecv);

        recv_tags.reserve(N_rcvs);

        Vector<Vector<std::size_t> > recv_offset(nrecv);
        Vector<std::size_t> offset;
        recv_size.reserve(nrecv);
        offset.reserve(nrecv);
        std::size_t TotalRcvsVolume = 0;
        for (int i = 0; i < nrecv; ++i) {
            std::size_t nbytes = 0;
            for (int imf = 0; imf < nmfs; ++imf) {
                if (cmds[imf]) {
                    auto const& tags = *(cmds[imf]->m_RcvTags);
                    auto it = tags.find(recv_from[i]);
                    if (it != tags.end()) {
                        for (auto const& cct : it->second) {
                            auto& dfab = (*mf[imf])[cct.dstIndex];
                            recv_offset[i].push_back(nbytes);
                            recv_tags.push_back({dfab.array(scomp[imf],ncomp[imf]),
                                                 makeArray4<T const>(nullptr,cct.dbox,ncomp[imf]),
                                                 cct.dbox, Dim3{0,0,0}});
                            nbytes += dfab.nBytes(cct.dbox,ncomp[imf]);
                        }
                    }
                }
            }

            std::size_t acd = ParallelDescriptor::alignof_comm_data(nbytes);
            nbytes = amrex::aligned_size(acd, nbytes); // so that nbytes are aligned

            // Also need to align the offset properly
            TotalRcvsVolume = amrex::aligned_size(std::max(alignof(T),acd), TotalRcvsVolume);

            offset.push_back(TotalRcvsVolume);
            TotalRcvsVolume += nbytes;

            recv_size.push_back(nbytes);
        }

        the_recv_data = static_cast<char*>(amrex::The_FA_Arena()->alloc(TotalRcvsVolume));

        int k = 0;
        for (int i = 0; i < nrecv; ++i) {
            char* p = the_recv_data + offset[i];
            const int rank = ParallelContext::global_to_local_rank(recv_from[i]);
            recv_reqs[i] = ParallelDescriptor::Arecv
                (p, recv_size[i], rank, SeqNum, comm).req();
            for (int j = 0, nj = recv_offset[i].size(); j < nj; ++j) {
                recv_tags[k++].sfab.p = (T const*)(p + recv_offset[i][j]);
            }
        }
    }

    char* the_send_data = nullptr;
    Vector<int> send_rank;
    Vector<char*> send_data;
    Vector<std::size_t> send_size;
    Vector<MPI_Request> send_reqs;
    if (N_snds > 0) {
        for (int imf = 0; imf < nmfs; ++imf) {
            if (cmds[imf]) {
                auto const& tags = *(cmds[imf]->m_SndTags);
                for (auto const& kv : tags) {
                    send_rank.push_back(kv.first);
                }
            }
        }
        amrex::RemoveDuplicates(send_rank);
        const int nsend = send_rank.size();

        send_data.resize(nsend, nullptr);
        send_reqs.resize(nsend, MPI_REQUEST_NULL);

        Vector<TagT> send_tags;
        send_tags.reserve(N_snds);

        Vector<Vector<std::size_t> > send_offset(nsend);
        Vector<std::size_t> offset;
        send_size.reserve(nsend);
        offset.reserve(nsend);
        std::size_t TotalSndsVolume = 0;
        for (int i = 0; i < nsend; ++i) {
            std::size_t nbytes = 0;
            for (int imf = 0; imf < nmfs; ++imf) {
                if (cmds[imf]) {
                    auto const& tags = *(cmds[imf]->m_SndTags);
                    auto it = tags.find(send_rank[i]);
                    if (it != tags.end()) {
                        for (auto const& cct : it->second) {
                            auto const& sfab = (*mf[imf])[cct.srcIndex];
                            send_offset[i].push_back(nbytes);
                            send_tags.push_back({amrex::makeArray4<T>(nullptr,cct.sbox,ncomp[imf]),
                                                 sfab.const_array(scomp[imf],ncomp[imf]),
                                                 cct.sbox, Dim3{0,0,0}});
                            nbytes += sfab.nBytes(cct.sbox,ncomp[imf]);
                        }
                    }
                }
            }

            std::size_t acd = ParallelDescriptor::alignof_comm_data(nbytes);
            nbytes = amrex::aligned_size(acd, nbytes); // so that bytes are aligned

            // Also need to align the offset properly
            TotalSndsVolume = amrex::aligned_size(std::max(alignof(T),acd), TotalSndsVolume);

            offset.push_back(TotalSndsVolume);
            TotalSndsVolume += nbytes;

            send_size.push_back(nbytes);
        }

        the_send_data = static_cast<char*>(amrex::The_FA_Arena()->alloc(TotalSndsVolume));
        int k = 0;
        for (int i = 0; i < nsend; ++i) {
            send_data[i] = the_send_data + offset[i];
            for (int j = 0, nj = send_offset[i].size(); j < nj; ++j) {
                send_tags[k++].dfab.p = (T*)(send_data[i] + send_offset[i][j]);
            }
        }

        detail::fbv_copy(send_tags);

        FabArray<FAB>::PostSnds(send_data, send_size, send_rank, send_reqs, SeqNum);
    }

#if !defined(AMREX_DEBUG)
    int recv_flag;
    ParallelDescriptor::Test(recv_reqs, recv_flag, recv_stat);
#endif

    if (N_locs > 0) {
        detail::fbv_copy(local_tags);
#if !defined(AMREX_DEBUG)
        ParallelDescriptor::Test(recv_reqs, recv_flag, recv_stat);
#endif
    }

    if (N_rcvs > 0) {
        ParallelDescriptor::Waitall(recv_reqs, recv_stat);
#ifdef AMREX_DEBUG
        if (!FabArrayBase::CheckRcvStats(recv_stat, recv_size, SeqNum)) {
            amrex::Abort("FillBoundary(vector) failed with wrong message size");
        }
#endif

        detail::fbv_copy(recv_tags);

        amrex::The_FA_Arena()->free(the_recv_data);
    }

    if (N_snds > 0) {
        Vector<MPI_Status> stats(send_reqs.size());
        ParallelDescriptor::Waitall(send_reqs, stats);
        amrex::The_FA_Arena()->free(the_send_data);
    }

#endif  // #ifdef AMREX_USE_MPI
#endif  // #if 1 #else
}

template <class MF>
std::enable_if_t<IsFabArray<MF>::value>
FillBoundary (Vector<MF*> const& mf, const Periodicity& a_period = Periodicity::NonPeriodic())
{
    Vector<int> scomp(mf.size(), 0);
    Vector<int> ncomp;
    Vector<IntVect> nghost;
    Vector<Periodicity> period(mf.size(), a_period);
    ncomp.reserve(mf.size());
    nghost.reserve(mf.size());
    for (auto const& x : mf) {
        ncomp.push_back(x->nComp());
        nghost.push_back(x->nGrowVect());
    }
    FillBoundary(mf, scomp, ncomp, nghost, period);
}













//#ifdef JORDI_DUMMY


/*********************************************
 *
 *           begin JORDI section
 *
 *********************************************/


template <class FAB>
template <class F, typename std::enable_if<IsBaseFab<F>::value,int>::type Z>
void
FabArray<FAB>::FillBoundary_finish (bool& comm_complete)
{
#ifdef AMREX_USE_MPI

    BL_PROFILE("FabArray::FillBoundary_finish()");

    if (!fbd) { n_filled = IntVect::TheZeroVector(); comm_complete = true; return; }

    const FB* TheFB = fbd->fb;
    const int N_rcvs = TheFB->m_RcvTags->size();
    if (N_rcvs > 0 && !(fbd->recvs_complete))
    {
        int actual_n_rcvs = N_rcvs - std::count(fbd->recv_data.begin(), fbd->recv_data.end(), nullptr);

	//printf("%d %d %d %d\n", fbd->sends_complete, fbd->recvs_complete, N_rcvs, actual_n_rcvs);
        //fflush(stdout);

        if (actual_n_rcvs > 0) {
            int test_recv_flag = 0;

            //ParallelDescriptor::Waitall(fbd->recv_reqs, fbd->recv_stat);
            //test_recv_flag = 1;

            ParallelDescriptor::Test(fbd->recv_reqs, test_recv_flag, fbd->recv_stat);
	    if (test_recv_flag)
            {
                fbd->recvs_complete = true;
            }
	    else
            {
                fbd->recvs_complete = false;
	        return;
            }
#ifdef AMREX_DEBUG
            if (!CheckRcvStats(fbd->recv_stat, fbd->recv_size, fbd->tag))
            {
                amrex::Abort("FillBoundary_finish failed with wrong message size");
            }
#endif
        }

	Vector<const CopyComTagsContainer*> recv_cctc(N_rcvs,nullptr);
        for (int k = 0; k < N_rcvs; k++)
        {
            if (fbd->recv_size[k] > 0)
            {
                auto const& cctc = TheFB->m_RcvTags->at(fbd->recv_from[k]);
                recv_cctc[k] = &cctc;
            }
        }

        bool is_thread_safe = TheFB->m_threadsafe_rcv;

#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10) )
            if (Gpu::inGraphRegion())
            {
                FB_unpack_recv_buffer_cuda_graph(*TheFB, fbd->scomp, fbd->ncomp,
                                                 fbd->recv_data, fbd->recv_size,
                                                 recv_cctc, is_thread_safe);
            }
            else
#endif
            {
                unpack_recv_buffer_gpu(*this, fbd->scomp, fbd->ncomp, fbd->recv_data, fbd->recv_size,
                                       recv_cctc, FabArrayBase::COPY, is_thread_safe);
            }
        }
        else
#endif
        {
            unpack_recv_buffer_cpu(*this, fbd->scomp, fbd->ncomp, fbd->recv_data, fbd->recv_size,
                                   recv_cctc, FabArrayBase::COPY, is_thread_safe);
        }

        if (fbd->the_recv_data)
        {
            amrex::The_FA_Arena()->free(fbd->the_recv_data);
            fbd->the_recv_data = nullptr;
        }
    }


    const int N_snds = TheFB->m_SndTags->size();
    if (N_snds > 0) {
        Vector<MPI_Status> stats(fbd->send_reqs.size());
	int test_send_flag = 0;

        //ParallelDescriptor::Waitall(fbd->send_reqs, stats);
	//int test_send_flag;
	
        ParallelDescriptor::Test(fbd->send_reqs, test_send_flag, fbd->send_stat);

        if (test_send_flag)
        {
            fbd->sends_complete = true;
        }
        else
        {
            fbd->sends_complete = false;
            return;
        }
        amrex::The_FA_Arena()->free(fbd->the_send_data);
        fbd->the_send_data = nullptr;
    }

    /* begin JORDI section */
    comm_complete = true;
    delete fbd;
    fbd = nullptr;
    //fbd.reset();
    /* end JORDI section */

#endif
}


template <class FAB>
template <class F, typename std::enable_if<IsBaseFab<F>::value,int>::type Z>
void
FabArray<FAB>::AsyncFillBoundary (int scomp, int ncomp, const IntVect& nghost,
                                  const Periodicity& period, bool cross,
                                  bool enforce_periodicity_only)
{
    BL_PROFILE("FabArray::AsyncFillBoundary()");

    AsyncFillBoundarySend(scomp, ncomp, nghost, period, cross, enforce_periodicity_only);
    AsyncFillBoundaryRecv(scomp, ncomp, nghost, period, cross, enforce_periodicity_only);
    return;

    /******************************** 
     * 
     * FBEP_nowait()
     *
     * *****************************/
{
    bool work_to_do;
    if (enforce_periodicity_only) {
        work_to_do = period.isAnyPeriodic();
    } else {
        work_to_do = nghost.max() > 0;
    }
    if (!work_to_do) { return; }

    const FB& TheFB = getFB(nghost, period, cross, enforce_periodicity_only);

    if (ParallelContext::NProcsSub() == 1)
    {
        //
        // There can only be local work to do.
        //
        int N_locs = (*TheFB.m_LocTags).size();
        if (N_locs == 0) return;
#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10))
            if (Gpu::inGraphRegion())
            {
                FB_local_copy_cuda_graph_1(TheFB, scomp, ncomp);
            }
            else
#endif
            {
                FB_local_copy_gpu(TheFB, scomp, ncomp);
            }
        }
        else
#endif
        {
            FB_local_copy_cpu(TheFB, scomp, ncomp);
        }

    }

#ifdef BL_USE_MPI

    //
    // Do this before prematurely exiting if running in parallel.
    // Otherwise sequence numbers will not match across MPI processes.
    //
    int SeqNum = 10;//ParallelDescriptor::SeqNum();

    const int N_locs = TheFB.m_LocTags->size();
    const int N_rcvs = TheFB.m_RcvTags->size();
    const int N_snds = TheFB.m_SndTags->size();

    if (N_locs == 0 && N_rcvs == 0 && N_snds == 0) {
        // No work to do.
        return;
    }

    async_fbd->fb    = &TheFB;
    async_fbd->scomp = scomp;
    async_fbd->ncomp = ncomp;
    async_fbd->nghost = nghost;
    async_fbd->period = period;
    async_fbd->cross = cross;
    async_fbd->epo   = enforce_periodicity_only;
    async_fbd->tag   = SeqNum;


    //
    // Post rcvs. Allocate one chunk of space to hold'm all.
    //

    if (N_rcvs > 0) {
        AsyncPostRcvs(*TheFB.m_RcvTags, async_fbd->the_recv_data,
                      async_fbd->recv_data, async_fbd->recv_size, async_fbd->recv_from, async_fbd->recv_reqs,
                      ncomp, SeqNum);
    }


    //
    // Post send's
    //
    char*&                          the_send_data = async_fbd->the_send_data;
    Vector<char*> &                     send_data = async_fbd->send_data;
    Vector<std::size_t>                 send_size = async_fbd->send_size;
    Vector<int>                         send_rank = async_fbd->send_rank;
    Vector<MPI_Request>&                send_reqs = async_fbd->send_reqs;
    Vector<const CopyComTagsContainer*> send_cctc = async_fbd->send_cctc;


    if (N_snds > 0)
    {
//#ifdef AMREX_USE_GPU
//        if (Gpu::inLaunchRegion())
//        {
//#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10))
//            if (Gpu::inGraphRegion()) {
//                FB_pack_send_buffer_cuda_graph(TheFB, scomp, ncomp, send_data, send_size, send_cctc);
//            }
//            else
//#endif
//            {
//                pack_send_buffer_gpu(*this, scomp, ncomp, send_data, send_size, send_cctc);
//            }
//        }
//        else
//#endif
//        {
//            pack_send_buffer_cpu(*this, scomp, ncomp, send_data, send_size, send_cctc);
//        }

        AMREX_ASSERT(send_reqs.size() == N_snds);
        AsyncPostSnds(send_data, send_size, send_rank, send_reqs, SeqNum);
    }

    //
    // Do the local work.  Hope for a bit of communication/computation overlap.
    //
    if (N_locs > 0)
    {
#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10) )
            if (Gpu::inGraphRegion()) {
                FB_local_copy_cuda_graph_n(TheFB, scomp, ncomp);
            }
            else
#endif
            {
                FB_local_copy_gpu(TheFB, scomp, ncomp);
            }
        }
        else
#endif
        {
            FB_local_copy_cpu(TheFB, scomp, ncomp);
        }
    }
}





    /********************************
     *
     * FillBoundary_finish()
     *
     * *****************************/

//{
//    const FB* TheFB = async_fbd->fb;
//    const int N_rcvs = TheFB->m_RcvTags->size();
//    if (N_rcvs > 0)
//    {
//	//ParallelDescriptor::Waitall(async_fbd->recv_reqs, async_fbd->recv_stat);
//        bool is_thread_safe = TheFB->m_threadsafe_rcv;
//#ifdef AMREX_USE_GPU
//        if (Gpu::inLaunchRegion())
//        {
//#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10) )
//            if (Gpu::inGraphRegion())
//            {
//                FB_unpack_recv_buffer_cuda_graph(*TheFB, async_fbd->scomp, async_fbd->ncomp,
//                                                 async_fbd->recv_data, async_fbd->recv_size,
//                                                 async_fbd->recv_cctc, is_thread_safe);
//            }
//            else
//#endif
//            {
//                unpack_recv_buffer_gpu(*this, async_fbd->scomp, async_fbd->ncomp, async_fbd->recv_data, async_fbd->recv_size,
//                                       async_fbd->recv_cctc, FabArrayBase::COPY, is_thread_safe);
//            }
//        }
//        else
//#endif
//        {
//            unpack_recv_buffer_cpu(*this, async_fbd->scomp, async_fbd->ncomp, async_fbd->recv_data, async_fbd->recv_size,
//                                   async_fbd->recv_cctc, FabArrayBase::COPY, is_thread_safe);
//        }
//    }
//
//    //ParallelDescriptor::Waitall(async_fbd->send_reqs, async_fbd->send_stat);
//
//}
#endif /*BL_USE_MPI*/
}


template <class FAB>
template <class F, typename std::enable_if<IsBaseFab<F>::value,int>::type Z>
void
FabArray<FAB>::AsyncFillBoundarySend (int scomp, int ncomp, const IntVect& nghost,
                                      const Periodicity& period, bool cross,
                                      bool enforce_periodicity_only)
{
    BL_PROFILE("FabArray::AsyncFillBoundarySend()");

    bool work_to_do;
    if (enforce_periodicity_only) {
        work_to_do = period.isAnyPeriodic();
    } else {
        work_to_do = nghost.max() > 0;
    }
    if (!work_to_do) { return; }

#ifdef BL_USE_MPI
    int SeqNum = 10;//ParallelDescriptor::SeqNum();
    async_fbd->tag = SeqNum;

    AsyncPostSnds(async_fbd->send_data, async_fbd->send_size, async_fbd->send_rank, async_fbd->send_reqs, SeqNum);
#endif /*BL_USE_MPI*/
}


template <class FAB>
template <class F, typename std::enable_if<IsBaseFab<F>::value,int>::type Z>
void
FabArray<FAB>::AsyncFillBoundaryRecv (int scomp, int ncomp, const IntVect& nghost,
                                      const Periodicity& period, bool cross,
                                      bool enforce_periodicity_only)
{
    BL_PROFILE("FabArray::AsyncFillBoundaryRecv()");

    bool work_to_do;
    if (enforce_periodicity_only) {
        work_to_do = period.isAnyPeriodic();
    } else {
        work_to_do = nghost.max() > 0;
    }
    if (!work_to_do) { return; }

    const FB& TheFB = getFB(nghost, period, cross, enforce_periodicity_only);

    if (ParallelContext::NProcsSub() == 1)
    {
        int N_locs = (*TheFB.m_LocTags).size();
        if (N_locs == 0) return;
#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10))
            if (Gpu::inGraphRegion())
            {
                FB_local_copy_cuda_graph_1(TheFB, scomp, ncomp);
            }
            else
#endif
            {
                FB_local_copy_gpu(TheFB, scomp, ncomp);
            }
        }
        else
#endif
        {
            FB_local_copy_cpu(TheFB, scomp, ncomp);
        }

    }

#ifdef BL_USE_MPI

    int SeqNum = 10;//ParallelDescriptor::SeqNum();

    const int N_locs = TheFB.m_LocTags->size();
    const int N_rcvs = TheFB.m_RcvTags->size();

//    if (N_locs > 0)
//    {
//#ifdef AMREX_USE_GPU
//        if (Gpu::inLaunchRegion())
//        {
//#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10) )
//            if (Gpu::inGraphRegion()) {
//                FB_local_copy_cuda_graph_n(TheFB, scomp, ncomp);
//            }
//            else
//#endif
//            {
//                FB_local_copy_gpu(TheFB, scomp, ncomp);
//            }
//        }
//        else
//#endif
//        {
//            FB_local_copy_cpu(TheFB, scomp, ncomp);
//        }
//    }

    AsyncPostRcvs(*TheFB.m_RcvTags, async_fbd->the_recv_data,
                  async_fbd->recv_data, async_fbd->recv_size, async_fbd->recv_from, async_fbd->recv_reqs,
                  ncomp, SeqNum);
#endif /*BL_USE_MPI*/
}


#ifdef BL_USE_MPI

template <class FAB>
AMREX_NODISCARD TheFaArenaPointer
FabArray<FAB>::AsyncPrepareSendBuffers (const MapOfCopyComTagContainers&     SndTags,
                                   Vector<char*>&                       send_data,
                                   Vector<std::size_t>&                 send_size,
                                   Vector<int>&                         send_rank,
                                   Vector<MPI_Request>&                 send_reqs,
                                   Vector<const CopyComTagsContainer*>& send_cctc,
                                   int                                  ncomp) const
{
    char* pointer = nullptr;
    PrepareSendBuffers(SndTags, pointer, send_data, send_size, send_rank, send_reqs, send_cctc, ncomp);
    return TheFaArenaPointer(pointer);
}

template <class FAB>
void
FabArray<FAB>::AsyncPrepareSendBuffers (const MapOfCopyComTagContainers&     SndTags,
                                   char*&                               the_send_data,
                                   Vector<char*>&                       send_data,
                                   Vector<std::size_t>&                 send_size,
                                   Vector<int>&                         send_rank,
                                   Vector<MPI_Request>&                 send_reqs,
                                   Vector<const CopyComTagsContainer*>& send_cctc,
                                   int                                  ncomp) const
{
    BL_PROFILE("FabArray::AsyncPrepareSendBuffers()");

    const int N_snds = SndTags.size();
    Vector<std::size_t> offset;
    std::size_t total_volume = 0;
    for (auto const& kv : SndTags)
    {
        Vector<int> iss;
        auto const& cctc = kv.second;

        std::size_t nbytes = 0;
        for (auto const& cct : kv.second)
        {
            nbytes += (*this)[cct.srcIndex].nBytes(cct.sbox,ncomp);
        }

        std::size_t acd = ParallelDescriptor::alignof_comm_data(nbytes);
        nbytes = amrex::aligned_size(acd, nbytes); // so that bytes are aligned

        // Also need to align the offset properly
        total_volume = amrex::aligned_size(std::max(alignof(typename FAB::value_type),
                                                    acd),
                                           total_volume);

        offset.push_back(total_volume);
        total_volume += nbytes;

        send_size.push_back(nbytes);
        send_rank.push_back(kv.first);
        send_cctc.push_back(&cctc);
        send_data.push_back(nullptr);
        send_reqs.push_back(MPI_REQUEST_NULL);
        async_fbd->send_count.push_back(0);
        async_fbd->send_finalMsg.push_back(0);
        async_fbd->send_flags.push_back(0);
    }

    int nsend = send_data.size(); 
    async_fbd->send_stat.resize(nsend);

    if (total_volume > 0)
    {
        the_send_data = static_cast<char*>(amrex::The_FA_Arena()->alloc(total_volume));
        for (int i = 0, N = send_size.size(); i < N; ++i) {
            send_data[i] = the_send_data + offset[i];
        }
    } else {
        the_send_data = nullptr;
    }
}


template <class FAB>
void
FabArray<FAB>::AsyncPostSnds (Vector<char*> const&       send_data,
                              Vector<std::size_t> const& send_size,
                              Vector<int> const&         send_rank,
                              Vector<MPI_Request>&       send_reqs,
                              int                        SeqNum) const
{
    BL_PROFILE("FabArray::AsyncPostSnds()");
    MPI_Comm comm = ParallelContext::CommunicatorSub();

    const int N_snds = send_data.size();
    int count_num_sends = 0;
    for (int j = 0; j < N_snds; ++j)
    {
        if (async_fbd->send_finalMsg[j] <= 1)
        {
            async_fbd->send_flags[j] = 0;
            ParallelDescriptor::Test(send_reqs[j], async_fbd->send_flags[j], async_fbd->send_stat[j]);
            if (async_fbd->send_flags[j] > 0) {
                count_num_sends++;
            }
        }
    }

    if (count_num_sends > 0)
    {
        //pack_send_buffer_gpu_subset(*this, async_fbd->scomp, async_fbd->ncomp, send_data, send_size, async_fbd->send_cctc, async_fbd->send_flags);
	const FB& TheFB = async_fbd->fb;

#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10))
            if (Gpu::inGraphRegion()) {
                FB_pack_send_buffer_cuda_graph(TheFB, async_fbd->scomp, async_fbd->ncomp, async_fbd->send_data, async_fbd->send_size, async_fbd->send_cctc);
            }
            else
#endif
            {
                pack_send_buffer_gpu(*this, async_fbd->scomp, async_fbd->ncomp, async_fbd->send_data, async_fbd->send_size, async_fbd->send_cctc);
            }
        }
        else
#endif
        {
            pack_send_buffer_cpu(*this, async_fbd->scomp, async_fbd->ncomp, async_fbd->send_data, async_fbd->send_size, async_fbd->send_cctc);
        }
    }

    for (int j = 0; j < N_snds; ++j)
    {
        if (async_fbd->send_finalMsg[j] <= 1)
        {
            if (async_fbd->send_flags[j] > 0) {
                int async_send_tag;
                if (async_fbd->send_finalMsg[j] == 1)
                {
                   async_send_tag = SeqNum+1;
                   async_fbd->send_finalMsg[j] = 2;
                }
                else
                {
                   async_send_tag = SeqNum;
                }
                const int rank = ParallelContext::global_to_local_rank(send_rank[j]);
                send_reqs[j] = ParallelDescriptor::Asend
                    (send_data[j], send_size[j], rank, async_send_tag, comm).req();
                async_fbd->send_count[j]++;
            }
        }
    }
}

template <class FAB>
TheFaArenaPointer FabArray<FAB>::AsyncPostRcvs (const MapOfCopyComTagContainers&       RcvTags,
                   Vector<char*>&                         recv_data,
                   Vector<std::size_t>&                   recv_size,
                   Vector<int>&                           recv_from,
                   Vector<MPI_Request>&                   recv_reqs,
                   int                                    ncomp,
                   int                                    SeqNum)
{
    char* pointer = nullptr;
    AsyncPostRcvs(RcvTags, pointer, recv_data, recv_size, recv_from, recv_reqs, ncomp, SeqNum);
    return TheFaArenaPointer(pointer);
}

template <class FAB>
void
FabArray<FAB>::AsyncPostRcvs (const MapOfCopyComTagContainers&  RcvTags,
                         char*&                            the_recv_data,
                         Vector<char*>&                    recv_data,
                         Vector<std::size_t>&              recv_size,
                         Vector<int>&                      recv_from,
                         Vector<MPI_Request>&              recv_reqs,
                         int                               ncomp,
                         int                               SeqNum)
{
    BL_PROFILE("FabArray::AsyncPostRcvs()");

    int nrecv = recv_data.size();
    MPI_Comm comm = ParallelContext::CommunicatorSub();
    int count_num_recvs = 0;

    Vector<int> at_least_one_recv(nrecv, 0);
    for (int i = 0; i < nrecv; ++i)
    {
        if (async_fbd->recvd_finalMsg[i] == 0)
        {
            const int rank = ParallelContext::global_to_local_rank(recv_from[i]);
            if (async_fbd->recv_count[i] == -1){
               recv_reqs[i] = ParallelDescriptor::Arecv
                   (async_fbd->recv_data_inFlight[i], recv_size[i], rank, SeqNum, comm).req();
               async_fbd->recv_reqs_finalMsg[i] = ParallelDescriptor::Arecv
                   (async_fbd->recv_data_finalMsg[i], recv_size[i], rank, SeqNum+1, comm).req();
               async_fbd->recv_count[i] = 0;
            }

            while (1)
            {
                async_fbd->recv_flags[i] = 0;
                ParallelDescriptor::Test(recv_reqs[i], async_fbd->recv_flags[i], async_fbd->recv_stat[i]);
                if (async_fbd->recv_flags[i] > 0)
                {
		    recv_data[i] = async_fbd->recv_data_inFlight[i];
                    recv_reqs[i] = ParallelDescriptor::Arecv
                        (async_fbd->recv_data_inFlight[i], recv_size[i], rank, SeqNum, comm).req();
                    async_fbd->recv_count[i]++;

		    at_least_one_recv[i] = 1;
		    count_num_recvs++;
                }
                else {
                    break;
                }
            }
    	
            async_fbd->recv_flags[i] = 0;
            ParallelDescriptor::Test(async_fbd->recv_reqs_finalMsg[i], async_fbd->recv_flags[i], async_fbd->recv_stat_finalMsg[i]);
            if (async_fbd->recv_flags[i] > 0)
            {
		recv_data[i] = async_fbd->recv_data_finalMsg[i];
                async_fbd->recvd_finalMsg[i] = 1;
                async_fbd->recv_count[i]++;
                MPI_Cancel(&(recv_reqs[i]));

		at_least_one_recv[i] = 1;
		count_num_recvs++;
            }

            //async_fbd->recv_flags[i] = 0;
            //ParallelDescriptor::Test(recv_reqs[i], async_fbd->recv_flags[i], async_fbd->recv_stat[i]);
            //if (async_fbd->recv_flags[i] > 0)
            //{
	    //    recv_data[i] = async_fbd->recv_data_inFlight[i];
            //    recv_reqs[i] = ParallelDescriptor::Arecv
            //        (async_fbd->recv_data_inFlight[i], recv_size[i], rank, SeqNum, comm).req();
            //    async_fbd->recv_count[i]++;
            //}
    	    //else 
    	    //{
            //    async_fbd->recv_flags[i] = 0;
            //    ParallelDescriptor::Test(async_fbd->recv_reqs_finalMsg[i], async_fbd->recv_flags[i], async_fbd->recv_stat_finalMsg[i]);
            //    if (async_fbd->recv_flags[i] > 0)
            //    {
	    //        recv_data[i] = async_fbd->recv_data_finalMsg[i];
            //        async_fbd->recvd_finalMsg[i] = 1;
            //        async_fbd->recv_count[i]++;
            //        MPI_Cancel(&(recv_reqs[i]));
            //    }
            //}
        }
    }

    if (count_num_recvs > 0)
    {
        const FB* TheFB = async_fbd->fb;
        const int N_locs = TheFB->m_LocTags->size();
        if (N_locs > 0)
        {
#ifdef AMREX_USE_GPU
            if (Gpu::inLaunchRegion())
            {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10) )
                if (Gpu::inGraphRegion()) {
                    FB_local_copy_cuda_graph_n(*TheFB, async_fbd->scomp, async_fbd->ncomp);
                }
                else
#endif
                {
                    FB_local_copy_gpu(*TheFB, async_fbd->scomp, async_fbd->ncomp);
                }
            }
            else
#endif
            {
                FB_local_copy_cpu(*TheFB, async_fbd->scomp, async_fbd->ncomp);
            }
        }
        bool is_thread_safe = TheFB->m_threadsafe_rcv;

        //unpack_recv_buffer_gpu_subset(*this, async_fbd->scomp, async_fbd->ncomp, async_fbd->recv_data, async_fbd->recv_size,
        //                       async_fbd->recv_cctc, FabArrayBase::COPY, is_thread_safe, at_least_one_recv);
        
#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10) )
            if (Gpu::inGraphRegion())
            {
                FB_unpack_recv_buffer_cuda_graph(*TheFB, async_fbd->scomp, async_fbd->ncomp,
                                                 async_fbd->recv_data, async_fbd->recv_size,
                                                 async_fbd->recv_cctc, is_thread_safe);
            }
            else
#endif
            {
                unpack_recv_buffer_gpu(*this, async_fbd->scomp, async_fbd->ncomp, async_fbd->recv_data, async_fbd->recv_size,
                                       async_fbd->recv_cctc, FabArrayBase::COPY, is_thread_safe);
            }
        }
        else
#endif
        {
            unpack_recv_buffer_cpu(*this, async_fbd->scomp, async_fbd->ncomp, async_fbd->recv_data, async_fbd->recv_size,
                                   async_fbd->recv_cctc, FabArrayBase::COPY, is_thread_safe);
        }
        
    }
}

template <class FAB>
int
FabArray<FAB>::AsyncCheckSendConverge (int num_msgs_per_neighb)
{
    int all_final_sends_posted = 1;
    int nsend = async_fbd->send_data.size();
    for (int i = 0; i < nsend; ++i)
    {
        if (async_fbd->send_finalMsg[i] == 0)
        {
            all_final_sends_posted = 0;
            if (async_fbd->send_count[i] >= num_msgs_per_neighb)
            {
                async_fbd->send_finalMsg[i] = 1;
            }
        }
        else
	{
            if (async_fbd->send_finalMsg[i] == 1)
            {
                all_final_sends_posted = 0;
            }
        }
    }
    return all_final_sends_posted;
}

template <class FAB>
int
FabArray<FAB>::AsyncCheckRecvConverge ()
{
    int nrecv = async_fbd->recv_data.size();
    for (int i = 0; i < nrecv; ++i)
    {
       if (async_fbd->recvd_finalMsg[i] == 0)
       {
           return 0;
       }
    }
    return 1;
}
#endif

template <class FAB>
int
FabArray<FAB>::AsyncCheckConverge (int extern_converge,
                                   int num_msgs_per_neighb)
{
    int converge_flag = 1;
#ifdef BL_USE_MPI
    int nsend = async_fbd->send_data.size();
    int nrecv = async_fbd->recv_data.size();
    
    if (extern_converge == 0){
        converge_flag = 0;
    }
    else
    {
        int recvd_enough = 1;
        for (int i = 0; i < nrecv; ++i)
        {
            if (async_fbd->recv_count[i] < num_msgs_per_neighb)
            {
                recvd_enough = 0;
		break;
            }
        }

	if (recvd_enough == 1)
        {
            int all_final_sends_posted = 1;
            for (int i = 0; i < nsend; ++i)
            {
                if (async_fbd->send_finalMsg[i] == 0)
                {
                    all_final_sends_posted = 0;
                    if (async_fbd->send_count[i] >= num_msgs_per_neighb)
                    {
                        async_fbd->send_finalMsg[i] = 1;
                    }
                }
                else
                {
                    if (async_fbd->send_finalMsg[i] == 1)
                    {
                        all_final_sends_posted = 0;
                    }
                }
            }
	    if (all_final_sends_posted == 0)
            {
                converge_flag = 0;
            }
        }
	else
	{
            converge_flag = 0;
	}
    }
#endif

    return converge_flag;
}

template <class FAB>
void
FabArray<FAB>::AsyncClearAllBuffers ()
{
   async_fbd->recv_flags.clear();
   async_fbd->send_flags.clear();

   async_fbd->recv_count.clear();
   async_fbd->send_count.clear();

   async_fbd->recv_data.clear();
   async_fbd->send_data.clear();

   async_fbd->recv_reqs.clear();
   async_fbd->send_reqs.clear();

   async_fbd->recv_stat.clear();
   async_fbd->send_stat.clear();

   async_fbd->recvd_finalMsg.clear();
   async_fbd->send_finalMsg.clear();

   async_fbd->recv_size.clear();
   async_fbd->send_size.clear();

   async_fbd->recv_from.clear();
   async_fbd->send_rank.clear();

   async_fbd->recv_cctc.clear();
   async_fbd->send_cctc.clear();

   async_fbd->recv_data_inFlight.clear();
   async_fbd->recv_data_finalMsg.clear();
   async_fbd->recv_stat_finalMsg.clear();
   async_fbd->recv_reqs_finalMsg.clear();

   if (async_fbd->the_recv_data)
   {
       amrex::The_FA_Arena()->free(async_fbd->the_recv_data);
       async_fbd->the_recv_data = nullptr;
   }
   if (async_fbd->the_send_data)
   {
       amrex::The_FA_Arena()->free(async_fbd->the_send_data);
       async_fbd->the_send_data = nullptr;
   }
}

template <class FAB>
void
FabArray<FAB>::AsyncSetup (int scomp, int ncomp, const IntVect& nghost,
                           const Periodicity& period, bool cross,
                           bool enforce_periodicity_only)
{
#ifdef BL_USE_MPI
   async_fbd = new FBData<FAB>();
   AsyncClearAllBuffers();



   const FB& TheFB = getFB(nghost, period, cross, enforce_periodicity_only);
   const MapOfCopyComTagContainers& RcvTags = *TheFB.m_RcvTags;
   Vector<std::size_t> offset;
   std::size_t TotalRcvsVolume = 0;
   for (const auto& kv : RcvTags)
   {
       std::size_t nbytes = 0;
       for (auto const& cct : kv.second)
       {
           nbytes += (*this)[cct.dstIndex].nBytes(cct.dbox,ncomp);
       }

       std::size_t acd = ParallelDescriptor::alignof_comm_data(nbytes);
       nbytes = amrex::aligned_size(acd, nbytes);

       TotalRcvsVolume = amrex::aligned_size(std::max(alignof(typename FAB::value_type),acd),
                                             TotalRcvsVolume);

       offset.push_back(TotalRcvsVolume);
       TotalRcvsVolume += nbytes;

       async_fbd->recv_size.push_back(nbytes);
       async_fbd->recv_from.push_back(kv.first);

       async_fbd->recv_data.push_back(nullptr);
       async_fbd->recv_reqs.push_back(MPI_REQUEST_NULL);
       async_fbd->recv_data_inFlight.push_back(nullptr);
       async_fbd->recv_data_finalMsg.push_back(nullptr);
       async_fbd->recv_flags.push_back(0);
       async_fbd->recvd_finalMsg.push_back(0);
       async_fbd->recv_count.push_back(-1);
       async_fbd->recv_reqs_finalMsg.push_back(MPI_REQUEST_NULL);
       async_fbd->recv_cctc.push_back(nullptr);
   }

   const int N_rcvs = TheFB.m_RcvTags->size();
   for (int k = 0; k < N_rcvs; k++)
   {
       if (async_fbd->recv_size[k] > 0)
       {
           auto const& cctc = TheFB.m_RcvTags->at(async_fbd->recv_from[k]);
           async_fbd->recv_cctc[k] = &cctc;
       }
   }

   int nrecv = async_fbd->recv_data.size();
   async_fbd->recv_stat.resize(nrecv);
   async_fbd->recv_stat_finalMsg.resize(nrecv);

   async_fbd->the_recv_data = static_cast<char*>(amrex::The_FA_Arena()->alloc(TotalRcvsVolume));
   for (int i = 0; i < nrecv; ++i)
   {
       async_fbd->the_recv_data[i] = 0.0;
   }

   for (int i = 0; i < nrecv; ++i)
   {
       async_fbd->recv_data[i] = async_fbd->the_recv_data + offset[i];
       async_fbd->recv_data_inFlight[i] = async_fbd->the_recv_data + offset[i];
       async_fbd->recv_data_finalMsg[i] = async_fbd->the_recv_data + offset[i];

       for (int j = 0; j < async_fbd->recv_size[i]; ++j)
       {
           async_fbd->recv_data[i][j] = 0.0;
           async_fbd->recv_data_inFlight[i][j] = 0.0;
           async_fbd->recv_data_finalMsg[i][j] = 0.0;
       }
   }

   AsyncPrepareSendBuffers(*TheFB.m_SndTags, async_fbd->the_send_data, async_fbd->send_data, async_fbd->send_size, async_fbd->send_rank,
                           async_fbd->send_reqs, async_fbd->send_cctc, ncomp);


   async_fbd->fb = &TheFB;
   async_fbd->scomp = scomp;
   async_fbd->ncomp = ncomp;
   async_fbd->nghost = nghost;
   async_fbd->period = period;
   async_fbd->cross = cross;
   async_fbd->epo = enforce_periodicity_only;
#endif
}

template <class FAB>
void
FabArray<FAB>::AsyncCleanup (int scomp, int ncomp, const IntVect& nghost,
                             const Periodicity& period, bool cross,
                             bool enforce_periodicity_only)
{
#ifdef BL_USE_MPI
   int SeqNum = 10;//ParallelDescriptor::SeqNum();
   const FB& TheFB = getFB(nghost, period, cross, enforce_periodicity_only);
   const int N_rcvs = async_fbd->recv_data.size();
   const int N_snds = async_fbd->send_data.size();
   MPI_Comm comm = ParallelContext::CommunicatorSub();
   //int all_final_sent_msgs_complete;
   if (N_rcvs > 0) {
       while(1) {
           AsyncPostRcvs(*TheFB.m_RcvTags, async_fbd->the_recv_data,
                         async_fbd->recv_data, async_fbd->recv_size, async_fbd->recv_from, async_fbd->recv_reqs,
                         ncomp, SeqNum);
           if (AsyncCheckRecvConverge())
           {
               break;
           }
       }
   }

   ParallelDescriptor::Waitall(async_fbd->send_reqs, async_fbd->send_stat);



   //printf("%d %d %d %d\n", N_snds, N_rcvs, 
   //       *std::min_element(async_fbd->recv_count.begin(), async_fbd->recv_count.end()),
   //       *std::min_element(async_fbd->send_count.begin(), async_fbd->send_count.end()));
   //for (int j = 0; j < N_rcvs; ++j) printf("%d %d\n", async_fbd->recv_count[j], async_fbd->send_count[j]);
   //for (int j = 0; j < N_snds; ++j) printf("%d\n", async_fbd->send_count[j]);

   AsyncClearAllBuffers();
   delete async_fbd;
#endif
}




/* end JORDI section */

//#endif
