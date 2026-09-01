/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_full_mesh.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H
#define MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H

// #define DEBUG_ON
#define DEBUG_CLOCK_ON

#ifdef DEBUG_CLOCK_ON
#define DebugClock(i) timePoint[i] = GetSystemCycle()
#else
#define DebugClock(i)
#endif


#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#include "kernel_operator.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/math/cumsum.h"
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_dispatch_v2_quant.h"
#include "../common/op_kernel/mc2_moe_context.h"
#include "moe_distribute_v2_base.h"
#include "check_winsize.h"
#if __has_include( "../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "shmem.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "utils/debug/asc_printf.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60
namespace MoeDistributeDispatchV2FullMeshImpl {
constexpr uint8_t BUFFER_NUM = 2;        // 多buf
constexpr uint32_t STATE_OFFSET = 32U;  // 状态空间偏移地址
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr uint8_t COMM_NUM = 2;  // 通信域大小
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL; // 64 + 320
constexpr uint64_t FLAG_FIELD_OFFSET = 768UL * 1024UL; // 384 * 2，0/1标识区偏移
constexpr uint64_t CUMSUM_CAL_OFFSET = 868UL * 1024UL; // 768 + 100
constexpr uint64_t CUMSUM_FLAG_OFFSET = 876UL * 1024UL; // 868 + 8
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint64_t SPLIT_BLOCK_COUNT = 128UL;  // 128 = SPLIT_BLOCK_SIZE / sizeof(float)
constexpr uint32_t SYNC_OFFSET = 3U * 1024U; // 核间软同步偏移地址
constexpr uint32_t EXPAND_IDX_INFO = 3U;  // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id
constexpr int32_t  MAX_UB_SIZE = 190 * 1024;
constexpr uint32_t COMPARE_COUNT_PER_BLOCK = 256 / sizeof(int32_t);
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_DATA_COUNT = 120U; // 120 = SPLIT_BLOCK_DATA_SIZE / sizeof(float)
constexpr uint32_t AIV_STATE_SIZE = 64U;
constexpr uint32_t SFFVALUE_SIZE = 64U;
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint32_t CUMSUM_MAX_CORE_NUM = 16U;
constexpr uint32_t RANK_LIST_NUM = 2U;
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t RUNPOS_CALCUMSUM = 2U;
constexpr uint32_t RUNPOS_CUMSUMFLAG = 3U;
constexpr uint32_t RUNPOS_ARRIVECNT = 4U;
constexpr uint8_t EP_WORLD_SIZE_IDX = 1;
constexpr uint8_t SHARE_RANK_NUM_IDX = 2;
constexpr uint8_t MOE_NUM_IDX = 3;
constexpr uint64_t CYCLES_PER_US = 50UL;
constexpr uint8_t UB_ALIGN_DATA_COUNT = 8U; // 8 = UB_ALIGN / sizeof(float) = UB_ALIGN / sizeof(int32_t)
constexpr uint32_t DURATION_OFFSET = sizeof(int64_t) / sizeof(int32_t);
constexpr uint32_t FLAG_OFFSET = STATE_OFFSET / sizeof(float);
constexpr uint32_t STAGE_DONE_STRIDE = 512U;
constexpr uint32_t PREPARE_THREADS_NUM = 1024U;
constexpr uint32_t CALC_CNT_THREADS_NUM = 1024U;
constexpr uint32_t SET_GM_VALUE_THREAD_NUM = 2048U;
constexpr uint32_t WARP_SIZE = 32U;
constexpr uint32_t MAX_WQE_CNT = 128U;
constexpr uint32_t BUILD_THREADS_NUM = MAX_WQE_CNT;
constexpr uint32_t WQE_THREADS_NUM = 128U;
constexpr uint32_t MAX_TASK_NUM_PER_LOOP = MAX_WQE_CNT;
constexpr uint32_t HEAD_READY_STRIDE = 512U;
// kfm FOR URMA
constexpr uint32_t MAX_WQE_NUM = 256U * 6U;
constexpr uint32_t MAX_JETTY_NUM = 16U;
constexpr uint64_t URMA_CTRL_GUARD = 64UL * 1024UL;
constexpr uint32_t totalShmemSize_ = 1024UL * 1024UL * 512; // 与test_aclnn_...里面attributes.local_mem_size = 1024UL * 1024UL * 512保持一致
constexpr uint32_t PORT_NUM = 1;
constexpr uint32_t JETTY_NUM = 1;
constexpr uint32_t headReadyFlag = 20260817U;
constexpr float STAGE_DONE_FLAG = 1.0F;

constexpr AscendC::CumSumConfig cumSumConfig{true, true, false};

#define TemplateMC2TypeFullmeshClass typename XType, typename ExpandXOutType, int32_t QuantMode, \
                                     bool IsSmoothScaleExist, bool IsNeedAllgather
#define TemplateMC2TypeFullmeshFunc XType, ExpandXOutType, QuantMode, IsSmoothScaleExist, IsNeedAllgather

using namespace AscendC;
using namespace Mc2Kernel;
using namespace MoeDistributeV2Base;
using namespace Mc2Aclnn;

__simt_vf__ __aicore__ LAUNCH_BOUND(PREPARE_THREADS_NUM) inline void simt_prepare_mapping(
        __ubuf__ int32_t* expertIds, 
        __ubuf__ int32_t* dstExpCnt,
        __ubuf__ int32_t* dstExpOffsetList,
        __ubuf__ int32_t* dstExpOrderedOffsetList, 
        __ubuf__ uint32_t* totalRemoteSendNum, 
        int32_t axisBS,
        int32_t axisK, 
        int32_t beginExpertId,
        int32_t expertNum,
        uint32_t moeExpertNumPerRank,
        uint32_t myRankId)
{
    auto tid = AscendC::Simt::GetThreadIdx<0>();



    for (int row = tid; row < axisBS*axisK; row += PREPARE_THREADS_NUM) {
        int ep_id = *(expertIds+row); //expertIds[row];
        if (ep_id >= beginExpertId && ep_id < expertNum + beginExpertId) {
            int cur_ep_id =  ep_id - beginExpertId;               // 目标 ep 在本AIV的编号
            int pre_cnt = atomicAdd(dstExpCnt + cur_ep_id, uint32_t(1));  // 计数+1
            // 写入偏移：第 dstExpCnt[ep_id] 行，第 ep_id 列
            // if (tid == 10) {
            //     AscendC::Simt::printf("pre_cnt %d", pre_cnt);
            // }
            dstExpOffsetList[cur_ep_id*axisBS + pre_cnt] = row;
        }
    }
    AscendC::Simt::ThreadBarrier();

    // for (uint32_t flatThreadTask = tid; flatThreadTask < expertNum * axisBS; flatThreadTask += PREPARE_THREADS_NUM) {
    //     const uint32_t curExpertId = flatThreadTask / axisBS;
    //     const uint32_t curTokenSlot = flatThreadTask - curExpertId * axisBS;
    //     const uint32_t curExpertSlotBase = curExpertId * axisBS;

    //     int32_t curExpertTokenCntI32 = dstExpCnt[curExpertId];
    //     uint32_t curExpertTokenCnt = 0;
    //     if (curExpertTokenCntI32 > 0) {
    //         curExpertTokenCnt = static_cast<uint32_t>(curExpertTokenCntI32);
    //         if (curExpertTokenCnt > axisBS) {
    //             curExpertTokenCnt = axisBS;
    //         }
    //     }
    //     if(flatThreadTask % (axisBS) == 0){
    //         atomicAdd(totalSendNum, curExpertTokenCnt);
    //     }
    //     // 非有效 token 区间只写尾部 sentinel。由于排序输出 rank 一定在 [0, curExpertTokenCnt)，
    //     // curTokenSlot >= curExpertTokenCnt 写 -1 不会和有效输出发生竞争。
    //     if (curTokenSlot >= curExpertTokenCnt) {
    //         // dstExpOrderedOffsetList[curExpertSlotBase + curTokenSlot] = -1;
    //         continue;
    //     }

    //     const int32_t curTokenOriginalOffset = dstExpOffsetList[curExpertSlotBase + curTokenSlot];

    //     // sortedSlotInExpert 计算：
    //     // 比 curTokenOriginalOffset 小的元素数量决定升序位置；
    //     uint32_t sortedSlotInExpert = 0;
    //     for (uint32_t compareSlot = 0; compareSlot < curExpertTokenCnt; ++compareSlot) {
    //         const int32_t compareTokenOriginalOffset = dstExpOffsetList[curExpertSlotBase + compareSlot];
    //         if (compareTokenOriginalOffset < curTokenOriginalOffset) {
    //             sortedSlotInExpert++;
    //         }
    //     }

    //     dstExpOrderedOffsetList[curExpertSlotBase + sortedSlotInExpert] = curTokenOriginalOffset;
    // }

    auto maxthreadId = expertNum * axisBS;
    for (uint32_t flatThreadTask = tid; flatThreadTask < maxthreadId; flatThreadTask += PREPARE_THREADS_NUM) {
        uint32_t curLocalExpertId = flatThreadTask / axisBS;
        uint32_t curExpertId = beginExpertId + curLocalExpertId;
        uint32_t curRankId = curExpertId / moeExpertNumPerRank;
        uint32_t curTokenSlot = flatThreadTask % axisBS;
        uint32_t offsetNum = dstExpCnt[curLocalExpertId];
        if(curTokenSlot >= offsetNum)
            continue;
        if(curTokenSlot == 0){
            uint32_t remoteRankTaskNum = (curRankId == myRankId) ? 0 : offsetNum;
            atomicAdd(totalRemoteSendNum, remoteRankTaskNum);
        }
        uint32_t curExpertSlotBase = curLocalExpertId * axisBS;
        int32_t curTokenOriginalOffset = dstExpOffsetList[flatThreadTask];
        uint32_t sortedSlotInExpert = 0;

        for (uint32_t compareSlot = 0; compareSlot < offsetNum; ++compareSlot) {
            int32_t compareTokenOriginalOffset = dstExpOffsetList[curExpertSlotBase + compareSlot];
            if (compareTokenOriginalOffset < curTokenOriginalOffset) {
                sortedSlotInExpert++;
            }
        }
        dstExpOrderedOffsetList[curExpertSlotBase + sortedSlotInExpert] = curTokenOriginalOffset;
    }

    // const uint32_t warpId = tid / WARP_SIZE;
    // const uint32_t laneId = tid % WARP_SIZE;
    // int32_t compareTokenOriginalOffset;
    // uint32_t offsetNum;
    // uint32_t curExpertSlotBase;
    // int32_t curTokenOriginalOffset;
    // auto maxthreadId = expertNum * axisBS;
    // uint32_t sortedSlotInExpert;
    // for (uint32_t curExpertId = warpId; curExpertId < expertNum; curExpertId += PREPARE_THREADS_NUM / WARP_SIZE) {
    //     offsetNum = dstExpCnt[curExpertId];
    //     curExpertSlotBase = curExpertId * axisBS;
    //     for (uint32_t curTokenSlot = laneId; curTokenSlot < offsetNum; curTokenSlot +=  WARP_SIZE) {
    //         if(curTokenSlot == 0){
    //             atomicAdd(totalSendNum, offsetNum);
    //         }
    //         curTokenOriginalOffset = dstExpOffsetList[curExpertSlotBase + curTokenSlot];
    //         sortedSlotInExpert = 0;
    //         for (uint32_t compareSlot = 0; compareSlot < offsetNum; ++compareSlot) {
    //             compareTokenOriginalOffset = dstExpOffsetList[curExpertSlotBase + compareSlot];
    //             if (compareTokenOriginalOffset < curTokenOriginalOffset) {
    //                 sortedSlotInExpert++;
    //             }
    //         }
    //         dstExpOrderedOffsetList[curExpertSlotBase + sortedSlotInExpert] = curTokenOriginalOffset;
    //     }
    //     // if(curTokenSlot >= offsetNum)
    //     //     continue;
    //     // if (curExpertId)
    // }
}



__simt_vf__ __aicore__ LAUNCH_BOUND(BUILD_THREADS_NUM) inline void buildRemoteWqeDesc(
        __ubuf__ uint32_t* expertCntList,       //输入，专家发送的Token个数信息
        __ubuf__ int32_t* expertOrderedOffsetList, //输入，按源 token 顺序排列的专家偏移
        uint32_t expertNum,                     //输入，expertCntList的有效长度
        uint32_t beginExpertId,                 //输入，起始专家Id
        uint32_t axisBS,                        //输入，expertOffsetList每个exp行的最大长度
        uint32_t axisK,                         //输入，Token发送给axisK个专家
        uint64_t srcBaseAddr,                   //输入，Token存放起始地址
        uint64_t tokenCommSize,                 //输入，Token传输长度，单位Byte
        uint32_t expertPerRank,                 //输入，单卡Moe专家数
        uint32_t localRankId,
        __ubuf__ uint64_t* remoteAddrList,      //输入，远端接收buffer地址存放的List
        uint32_t opcode,
        __ubuf__ uint64_t* localAddrList,
        __ubuf__ uint64_t* remoteWqeAddrList,
        __ubuf__ uint32_t* dstRankList,
        __ubuf__ uint32_t* messageLenList,
        __ubuf__ uint32_t* opcodeList,
        uint32_t loopIdx,
        __ubuf__ uint32_t* finishedWqeDescNum
        )
{
    const uint32_t tid = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t warpId = tid / WARP_SIZE;
    const uint32_t laneId = tid % WARP_SIZE;

    uint32_t currBlock = (BUILD_THREADS_NUM / WARP_SIZE) * loopIdx + warpId;
    uint32_t targetLocalExp = currBlock % expertNum;
    uint32_t currBatchOfCurrExp = currBlock / expertNum;
    uint32_t mySlot = currBatchOfCurrExp * WARP_SIZE + laneId;
    uint32_t targetExpId = targetLocalExp + beginExpertId;
    uint32_t targetRankId = targetExpId / expertPerRank;
    uint32_t validOffsetCnt = expertCntList[targetLocalExp];
    if (targetRankId != localRankId && mySlot < validOffsetCnt) {
        int32_t offsetValue = expertOrderedOffsetList[targetLocalExp * axisBS + mySlot];
        uint32_t targetExpertOnRank = targetExpId % expertPerRank;
        uint32_t beginRankId = beginExpertId / expertPerRank;
        uint32_t tokenId = uint32_t(offsetValue) / axisK;
        uint64_t remoteBaseAddr = remoteAddrList[targetRankId - beginRankId];
        uint32_t taskOffset = atomicAdd(finishedWqeDescNum, 1);
        localAddrList[taskOffset] = srcBaseAddr + tokenId * tokenCommSize;
        remoteWqeAddrList[taskOffset] =
            remoteBaseAddr + (targetExpertOnRank * axisBS + mySlot) * tokenCommSize;
        dstRankList[taskOffset] = targetRankId;
        messageLenList[taskOffset] = tokenCommSize;
        opcodeList[taskOffset] = opcode;
    }

    AscendC::Simt::ThreadBarrier();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CALC_CNT_THREADS_NUM) inline void calc_rank_token_cnt(
    __ubuf__ int32_t* expertIds,
    __ubuf__ uint32_t* remoteRankTokenCnt,
    uint32_t expertIdsSize,
    uint32_t moeExpertNumPerRank,
    __ubuf__ uint32_t* remoteRanksNum,
    uint32_t myRankId,
    uint32_t epWorldSize
) {
    const uint32_t tid = AscendC::Simt::GetThreadIdx<0>();
    
    for (uint32_t rankIdx = tid; rankIdx < epWorldSize; rankIdx += CALC_CNT_THREADS_NUM) {
        remoteRankTokenCnt[rankIdx] = 0;
    }
    AscendC::Simt::ThreadBarrier();
    for (uint32_t slot = tid; slot < expertIdsSize; slot += CALC_CNT_THREADS_NUM) {
        int32_t expertId = expertIds[slot];
        if (expertId < 0) {
            continue;
        }
        uint32_t rankId = static_cast<uint32_t>(expertId) / moeExpertNumPerRank;
        if (rankId < epWorldSize && rankId != myRankId) {
            atomicAdd(remoteRankTokenCnt + rankId, 1U);
        }
    }
    AscendC::Simt::ThreadBarrier();
    for (uint32_t rankIdx = tid; rankIdx < epWorldSize; rankIdx += CALC_CNT_THREADS_NUM) {
        if (rankIdx != myRankId && remoteRankTokenCnt[rankIdx] != 0U) {
            atomicAdd(remoteRanksNum, 1U);
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SET_GM_VALUE_THREAD_NUM) inline void set_gm_value(
    __gm__ uint32_t* baseAddr,
    uint32_t totalNum,
    uint32_t stride,
    uint32_t flag
) {
    const uint32_t tid = AscendC::Simt::GetThreadIdx<0>();

    for (uint32_t slot = tid; slot < totalNum; slot += SET_GM_VALUE_THREAD_NUM) {
        baseAddr[slot * stride / sizeof(uint32_t)] = flag;
    }
    // asc_threadfence();
    AscendC::Simt::ThreadBarrier();
}

template <TemplateMC2TypeFullmeshClass>
class MoeDistributeDispatchV2FullMesh {
public:
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    using XInType = typename std::conditional<
            (Std::IsSame<XType, fp4x2_e2m1_t>::value) || (Std::IsSame<XType, fp4x2_e1m2_t>::value),
            uint8_t,
            XType
        >::type;
    using XOutType = typename std::conditional<
            (Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value) || (Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value),
            uint8_t,
            ExpandXOutType
        >::type;
#else
    using XInType = XType;
    using XOutType = ExpandXOutType;
#endif
    __aicore__ inline MoeDistributeDispatchV2FullMesh() {};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo, GM_ADDR performanceInfo,
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ExpIdsCopyAndMaskCal();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void InitElasticInfo();
    __aicore__ inline void SetDataStatus();
    __aicore__ inline void ClearStageDoneFlags();
    __aicore__ inline void PublishStageDone();
    __aicore__ inline void WaitAllStageDone(LocalTensor<float> stageDoneTensor,
        LocalTensor<float> stageDoneWorkTensor, LocalTensor<float> stageDoneSumTensor);
    // __aicore__ inline void InitHeadRecord();
    __aicore__ inline void SetTilingData(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void CalValidBSCnt(LocalTensor<bool> maskStrideTensor);
    __aicore__ inline void CalValidExpIdx(LocalTensor<bool> maskInputTensor);
    __aicore__ inline void GenerateGatherMaskTensor(uint32_t maskCnt);
    __aicore__ inline void MaskZeroComputeExpert(uint32_t maskCnt);
    __aicore__ inline void ZeroComputeExpertMaskCal();
    __aicore__ inline void SetTilingDataAndCal(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void SendToSharedExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf);
    __aicore__ inline void SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> expertMaskBuf, TBuf<> outBuf);
    __aicore__ inline void SendToMoeExpertNew(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> expertMaskBuf, TBuf<> outBuf);
    __aicore__ inline void CalExpertSendNum(TBuf<> outBuf, TBuf<> expertMaskBuf);
    __aicore__ inline void ExpertActiveMaskInit();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void AllToAllDispatch();
    __aicore__ inline void URMASendToken();
    __aicore__ inline void SelfCopyToken();
    __aicore__ inline void RingDoorbell();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void CalAndSendCnt();
    __aicore__ inline void BufferInit();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float> &gatherMaskOutTensor,
         LocalTensor<uint32_t> &gatherTmpTensor, LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId);
    __aicore__ inline void RunPosRecord(const uint32_t runPos);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum);
    __aicore__ inline uint32_t CheckDataArriveWithFlag(uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt);
    __aicore__ inline void CopyInAndOut(LocalTensor<int32_t> xOutInt32Tensor,
                                        GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount, uint32_t dstExpertId);
    __aicore__ inline void WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum);
    __aicore__ inline void SetExpertTokenNums();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void SplitExpertNumToCore(uint32_t &delCurExpertGroupNum, uint32_t &groupIdx);
    __aicore__ inline void FillTriple(LocalTensor<XOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void TokenToExpertInQuant(GlobalTensor<XOutType> dstWinGMTensor,
            TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex, uint32_t toExpertId, uint32_t toExpertIndex);
    __aicore__ inline void TokenToExpert(GlobalTensor<XOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
                                        uint32_t srcTokenIndex, uint32_t toExpertIndex, uint32_t toRankId = 0,
                                        uint32_t dstExpertId = 0, uint32_t dstTokenPreCnt = 0);
    __aicore__ inline void RecordRankCommDuration(LocalTensor<int32_t> &performanceInfoTensor, uint64_t startTime);
    __aicore__ inline void get_sqs_and_cqs_xb(GM_ADDR sqs, GM_ADDR cqs); 
    __aicore__ inline uint64_t GetHeadRecordRegionSize()
    {
        uint32_t N32numPerJetty = Ceil(axisBS_ * axisK_, 32U);
        uint64_t headRecordBytes = totalJettyNum *
            (N32numPerJetty * sizeof(uint32_t) + UB_ALIGN);
        return Ceil(headRecordBytes, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    }

    __aicore__ inline uint64_t GetUrmaCtrlReserveSize()
    {
        uint64_t ctrlBytes = aivUsedRemoteWqe_ * HEAD_READY_STRIDE +
            GetHeadRecordRegionSize() +
            totalJettyNum * (sizeof(HcclAiRMAWQ) + sizeof(HcclAiRMACQ));
        return Ceil(ctrlBytes, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN + URMA_CTRL_GUARD;
    }

    __aicore__ inline uint64_t GetStageDoneRegionSize()
    {
        return static_cast<uint64_t>(axisBS_) * STAGE_DONE_STRIDE;
    }

    __aicore__ inline uint32_t GetActiveStageNum()
    {
        return axisBS_ < aivUsedStage_ ? axisBS_ : aivUsedStage_;
    }

    __aicore__ inline uint32_t GetActiveMoeStageNum()
    {
        return axisBS_ < moeUsedAivNum_ ? axisBS_ : moeUsedAivNum_;
    }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        uint64_t dataBaseOffset =
            GetStageDoneRegionSize() +
            axisBS_ * hCommuSize_ +
            GetUrmaCtrlReserveSize() +
            FLAG_FIELD_OFFSET +
            aivNum_ * WIN_ADDR_ALIGN;
        
        // AscendC::printf("[kfm] GetWindAddrByRankId rankId %d aivId_ %d shmemBuffer start %p dataBaseOffset %d dataState_ %d\n", rankId, aivId_, aclshmem_ptr((GM_ADDR)shmemBuffer_, rankId),
        //     dataBaseOffset, dataState_);

        if (isMc2Context_) {
            return (GM_ADDR)((uint64_t)aclshmem_ptr((GM_ADDR)shmemBuffer_, rankId) + dataBaseOffset + shmemDataSizeOffset_);
            // return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffset_;
        }
        return GetBaseWindAddrByRankId(winContext_[COMM_EP_IDX], rankId, epRankIdOriginal_) + winDataSizeOffset_;
    }

    __aicore__ inline GM_ADDR GetWindStateBaseAddrByRankId(const int32_t rankId)
    {
    if (isMc2Context_) {
        return (GM_ADDR)((uint64_t)aclshmem_ptr((GM_ADDR)shmemBuffer_, rankId) +
            GetStageDoneRegionSize() +
            axisBS_ * hCommuSize_ +
            GetUrmaCtrlReserveSize());
    }
    return GetBaseWindStateAddrByRankId(winContext_[COMM_EP_IDX], rankId, epRankIdOriginal_);
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
    return GetWindStateBaseAddrByRankId(rankId) + dataState_ * WIN_STATE_OFFSET;
    }

    // The local shmem area starts with per-stage-core completion flags,
    // followed by the token staging buffer.
    __aicore__ inline GM_ADDR GetSendBufferAddrByTokenId(const int32_t tokenId)
    {
        // token send buffer 每个 token 占用 hCommuSize_ 字节
        if (isMc2Context_) {
            return (GM_ADDR)shmemBuffer_ +
            // WIN_STATE_OFFSET * 2 +
            GetStageDoneRegionSize() + tokenId * hCommuSize_;
            // return (GM_ADDR)mc2Context_->epHcclBuffer_[epRankId_] + 
            // WIN_STATE_OFFSET * 2 +
            // GetStageDoneRegionSize() + tokenId * hCommuSize_;
        }
        return GetBaseWindOutAddrByRankId(winContext_[COMM_EP_IDX], epRankIdOriginal_) +
            GetStageDoneRegionSize() + tokenId * hCommuSize_;
    }

    __aicore__ inline GM_ADDR GetStageDoneAddrByAivId(const int32_t stageAivId)
    {
        // Each stage core owns a separate cache-line-aligned completion flag.
        if (isMc2Context_) {
            return (GM_ADDR)shmemBuffer_ + 
            stageAivId * STAGE_DONE_STRIDE;
        }
        return GetBaseWindOutAddrByRankId(winContext_[COMM_EP_IDX], epRankIdOriginal_) +
            stageAivId * STAGE_DONE_STRIDE;
    }
    

    __aicore__ inline GM_ADDR GetHeadReadyAddr() {
        // return (GM_ADDR)((uint64_t)aclshmem_ptr((GM_ADDR)shmemBuffer_, epRankId_) + 
        //     GetStageDoneRegionSize() + axisBS_ * hCommuSize_);
        return (GM_ADDR)shmemBuffer_ + GetStageDoneRegionSize() + axisBS_ * hCommuSize_;
    }

    __aicore__ inline GM_ADDR GetHeadRecordAddr() {
        // return (GM_ADDR)((uint64_t)aclshmem_ptr((GM_ADDR)shmemBuffer_, epRankId_) + 
        //     GetStageDoneRegionSize() + axisBS_ * hCommuSize_);
        return GetHeadReadyAddr() + aivUsedRemoteWqe_ * HEAD_READY_STRIDE;
    }

    __aicore__ inline GM_ADDR GetSqsAddr() {
        // return (GM_ADDR)((uint64_t)aclshmem_ptr((GM_ADDR)shmemBuffer_, epRankId_) + 
        //     GetStageDoneRegionSize() + axisBS_ * hCommuSize_);
        return GetHeadRecordAddr() + GetHeadRecordRegionSize();
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XInType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<XOutType> winTpGatherOutGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<int32_t> performanceInfoGMTensor_;
    GlobalTensor<float> selfRankWinInGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> validExpertIdsTensor_;
    LocalTensor<int32_t> tokenNumToExpertTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<int32_t> performanceInfoTensor_;
    LocalTensor<int32_t> performanceFlagTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<float> cumSumTime1Tensor_;
    LocalTensor<float> cumSumTime2Tensor_;
    LocalTensor<float> tempTime1Tensor_;
    LocalTensor<float> tempTime2Tensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> statusSumOutTensor_;
    LocalTensor<int32_t> cleanStatusTensor_;
    LocalTensor<uint32_t> gatherTmpTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<uint8_t> sharedTmpBufTensor_;
    LocalTensor<float> syncOnCoreTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<float> statusCleanFp32Tensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<XOutType> outTensor_;
    LocalTensor<XOutType> tempTensor_;
    LocalTensor<float> floatLocalAbsTemp_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<uint8_t> flagCompResultU8_;
    LocalTensor<uint64_t> flagCompResultLtU64_;
    LocalTensor<uint32_t> flagRecvGatherMask_;
    LocalTensor<float> cleanUpTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    LocalTensor<XOutType> xTmpTensor_;
    LocalTensor<float> flagGatherOutTensor_;
    LocalTensor<float> flagRecvTensor_;
    LocalTensor<int32_t> clockResultLT;

    TBuf<> statusBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> elasticInfoBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> gatherMaskOutBuf_;
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_;
    TBuf<> validExpertIndexBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> performanceInfoBuf_;
    TBuf<> performanceFlagBuf_;
    TBuf<> clockResultBuf_;

    GM_ADDR expandXOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR sendTpCountOutGM_;
    GM_ADDR statusSpaceGM_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    GM_ADDR statusDataSpaceGM_;
    GM_ADDR statusBaseGM;
    
    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    int32_t epRankId_{0};
    int32_t epRankIdOriginal_{0};
    uint32_t aivId_{0};           // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};     // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0};  // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertRankNum_{0};  // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t totalExpertNum_{0};
    uint32_t dealRankPerCore_{0};
    uint32_t hOutSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t hAlignSize_{0};
    uint32_t hSizeAlignBlock_{0};
    uint32_t hOutSizeAlignBlock_{0};
    uint32_t scaleInBytes_{0};
    uint32_t scaleOutBytes_{0};
    uint32_t scalesCount_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t startId_;
    uint32_t endId_;
    uint32_t sendNum_;
    uint32_t statusCntAlign_;
    uint32_t dataState_{0};
    uint32_t tBufRealSize_{0};
    uint64_t shmemDataSizeOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    uint64_t flagPadOffset_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isPerformanceFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool isScalingDownFlag_ = false;
    bool isMc2Context_ = false;
    uint64_t totalWinSize_{0};
    uint32_t gatherCount_{0};
    uint32_t expertTokenNumsType_{1};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    int32_t zeroComputeExpertNum_{0};
    uint32_t axisHExpandXAlignSize_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t axisHCommu_{0};
    uint32_t hCommuSize_{0};
    uint32_t maskSizePerExpert_{0};
    uint32_t expertIdsBufSize_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint64_t cumSumUB_{0};
    uint32_t cumSumTimes_{0};
    uint32_t delLastExpertId_{0};
    uint32_t remainderExpertNum_{0};
    uint32_t aivUsedCumSum_{0};
    uint32_t aivUsedStage_{0};
    uint32_t aivUsedRemoteWqe_{0};
    uint32_t aivUsedSelfCopy_{4};
    uint32_t aivUsedDoorbell_{1};
    uint32_t aivCumSumStart_{0};
    uint32_t maxSize_{0};
    uint32_t expertIdsSize_{0};
    uint32_t globalBS_{0};
    uint64_t shmemBuffer_{0};
    uint32_t copyInAxisH_{0};
    uint32_t copyOutAxisH_{0};
    uint32_t totalJettyNum{0};
    __gm__ HcclOpParam *winContext_[COMM_NUM]{nullptr, nullptr};
    __gm__ Mc2MoeContext* mc2Context_{nullptr};

    DataCopyParams hCopyParams_;
    DataCopyParams dataStateParams_{1U, sizeof(uint32_t), 0U, 0U};

    MoeDistributeDispatchV2Quant<XInType, ExpandXOutType,
                XOutType, QuantMode, IsSmoothScaleExist, IsNeedAllgather> quantInst_;

#ifdef DEBUG_CLOCK_ON
    uint64_t timePoint[16];
#endif
};

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::InitElasticInfo()
{
    uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t);
    uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
    elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
    DataCopyExtParams elasticInfoParams = {1U, static_cast<uint32_t>((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor_, elasticInfoParams, elasticInfoCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    isScalingDownFlag_ = elasticInfoTensor_.GetValue(0);
    if (isScalingDownFlag_) {
        epWorldSize_ = elasticInfoTensor_.GetValue(EP_WORLD_SIZE_IDX);
        sharedExpertRankNum_ = elasticInfoTensor_.GetValue(SHARE_RANK_NUM_IDX);
        moeExpertNum_ = elasticInfoTensor_.GetValue(MOE_NUM_IDX);
        epRankId_ = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epRankId_);
    } 
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SetTilingData(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    epWorldSizeOriginal_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    epRankIdOriginal_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    hasElasticInfoFlag_ = tilingData->moeDistributeDispatchV2Info.hasElasticInfo;
    isPerformanceFlag_ = tilingData->moeDistributeDispatchV2Info.isPerformance;
    epRankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    globalBS_ = tilingData->moeDistributeDispatchV2Info.globalBs;
    epWorldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    zeroComputeExpertNum_ = tilingData->moeDistributeDispatchV2Info.zeroComputeExpertNum;
    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isExpertMask;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    scaleInBytes_ = tilingData->moeDistributeDispatchV2Info.scalesCol * \
                    tilingData->moeDistributeDispatchV2Info.scalesTypeSize;
    scalesCount_ = tilingData->moeDistributeDispatchV2Info.scalesCount;
    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    maxSize_ = tilingData->moeDistributeDispatchV2Info.maxSizeForUbBuffer;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SetTilingDataAndCal(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    SetTilingData(tilingData);
    copyInAxisH_ = axisH_;
    copyOutAxisH_ = axisH_;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if constexpr (Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value ||
        Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value) {
        copyOutAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
    }
    if constexpr (Std::IsSame<XType, fp4x2_e2m1_t>::value ||
        Std::IsSame<XType, fp4x2_e1m2_t>::value) {
        copyInAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
    }
#endif
    if (hasElasticInfoFlag_) {
        InitElasticInfo();
    }
    isShareExpertRankFlag_ = (epRankId_ < sharedExpertRankNum_);
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    expertIdsCnt_ = axisBS_ * axisK_;
    hOutSize_ = copyOutAxisH_ * sizeof(XOutType);
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, 
                         tokenQuantAlign_, hOutSizeAlign_, scaleOutBytes_, axisH_);
    // hzy: 三元组后追加完整 top-k expert id 列表，URMA 转发后用于恢复 expandIdx 的原始 top-k slot。
    uint32_t expandInfoSize = UB_ALIGN + Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    hAlignSize_ += expandInfoSize;
    hSizeAlignBlock_ = Ceil(hAlignSize_, SPLIT_BLOCK_DATA_COUNT) * SPLIT_BLOCK_DATA_COUNT;
    hOutSizeAlign_ = tokenQuantAlign_ * sizeof(int32_t) + expandInfoSize;
    blockCntPerToken_ = Ceil(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE);
    hOutSizeAlignBlock_ = blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE;
    hCommuSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    axisHCommu_ = hCommuSize_ / sizeof(XOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hCommuSize_;
    rscvStatusNum_ = isShareExpertRankFlag_ ? epWorldSize_ : (epWorldSize_ * moeExpertNumPerRank_);
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    statusCntAlign_ = Ceil(totalExpertNum_, UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;
    aivUsedCumSum_ = totalExpertNum_ / 16; // 单核处理32个专家cnt发送 // kfm
    aivUsedCumSum_ = (aivUsedCumSum_ == 0) ? 1 : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= (aivNum_ / 2)) ? (aivNum_ / 2) : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= CUMSUM_MAX_CORE_NUM) ? CUMSUM_MAX_CORE_NUM : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= rscvStatusNum_) ? rscvStatusNum_ : aivUsedCumSum_; // 确保每个核至少处理一个状态
    aivCumSumStart_ = aivNum_ - aivUsedCumSum_;
    aivUsedStage_ = aivCumSumStart_ - aivUsedDoorbell_;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivUsedStage_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    moeUsedAivNum_ = aivUsedStage_ - sharedUsedAivNum_;
    aivUsedRemoteWqe_ = aivUsedStage_ - aivUsedSelfCopy_;
    totalJettyNum = epWorldSize_ * PORT_NUM * JETTY_NUM;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::ClearStageDoneFlags()
{
    if (aivId_ >= GetActiveStageNum()) {
        return;
    }

    TBuf<> cleanStageDoneBuf;
    tpipe_->InitBuffer(cleanStageDoneBuf, UB_ALIGN);
    LocalTensor<float> cleanStageDoneTensor = cleanStageDoneBuf.Get<float>();
    Duplicate<float>(cleanStageDoneTensor, 0.0F, UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::V_MTE3>();

    GlobalTensor<float> stageDoneTensor;
    stageDoneTensor.SetGlobalBuffer((__gm__ float*)GetStageDoneAddrByAivId(aivId_));
    DataCopy(stageDoneTensor, cleanStageDoneTensor, UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::PublishStageDone()
{
    if (aivId_ >= GetActiveStageNum()) {
        return;
    }

    TBuf<> stageDoneBuf;
    tpipe_->InitBuffer(stageDoneBuf, UB_ALIGN);
    LocalTensor<float> stageDoneTensor = stageDoneBuf.Get<float>();
    Duplicate<float>(stageDoneTensor, 0.0F, UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::V_S>();
    stageDoneTensor.SetValue(0, STAGE_DONE_FLAG);
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    GlobalTensor<float> stageDoneGT;
    stageDoneGT.SetGlobalBuffer((__gm__ float*)GetStageDoneAddrByAivId(aivId_));
    DataCopy(stageDoneGT, stageDoneTensor, UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::WaitAllStageDone(
    LocalTensor<float> stageDoneTensor, LocalTensor<float> stageDoneWorkTensor,
    LocalTensor<float> stageDoneSumTensor)
{
    uint32_t activeStageNum = GetActiveStageNum();
    GlobalTensor<float> stageDoneGT;
    stageDoneGT.SetGlobalBuffer((__gm__ float*)GetStageDoneAddrByAivId(0));
    DataCopyParams stageDoneCopyParams{
        static_cast<uint16_t>(activeStageNum), 1,
        static_cast<uint16_t>(STAGE_DONE_STRIDE / UB_ALIGN - 1), 0};

    const float compareTarget = static_cast<float>(activeStageNum);
    float sumOfFlag = -1.0F;
    while (sumOfFlag != compareTarget) {
        DataCopy(stageDoneTensor, stageDoneGT, stageDoneCopyParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(stageDoneSumTensor, stageDoneTensor, stageDoneWorkTensor, 1,
            activeStageNum, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = stageDoneSumTensor.GetValue(0);
    }
}

// template <TemplateMC2TypeFullmeshClass>
// __aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::InitHeadRecord()
// {
    
//     uint32_t startJettyId, endJettyId, jettyNumPerCore;
//     SplitToCore(totalJettyNum, aivNum_, startJettyId, endJettyId, jettyNumPerCore, true);

//     if (startJettyId >= totalJettyNum) {
//         return;
//     }

//     TBuf<> cleanRecordBuf;
//     uint32_t headRecordSizePerJetty = Ceil(axisBS_ * axisK_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
//     tpipe_->InitBuffer(cleanRecordBuf, headRecordSizePerJetty);
//     LocalTensor<uint32_t> cleanRecordTensor = cleanRecordBuf.Get<uint32_t>();
//     Duplicate<uint32_t>(cleanRecordTensor, 0U, headRecordSizePerJetty / sizeof(uint32_t));
//     SyncFunc<AscendC::HardEvent::V_MTE3>();

//     GlobalTensor<uint32_t> headRecordTensor;
//     for (uint32_t jettyId = startJettyId; jettyId < endJettyId; ++jettyId) {
//         headRecordTensor.SetGlobalBuffer((__gm__ uint32_t*)(GetHeadRecordAddr() + jettyId * headRecordSizePerJetty));
//         DataCopy(headRecordTensor, cleanRecordTensor, headRecordSizePerJetty / sizeof(uint32_t));
//     }
//     SyncFunc<AscendC::HardEvent::MTE3_S>();
// }

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SetDataStatus()
{
    uint32_t epRankIdHccl{0};
    uint32_t epWorldSizeHccl{0};
    GM_ADDR statusFlagBaseGM{0};
    if (isMc2Context_) {
        // epRankIdHccl = mc2Context_->epRankId;
        epRankIdHccl = epRankId_;
        // statusDataSpaceGM_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankIdHccl]);
        statusFlagBaseGM = GetWindStateBaseAddrByRankId(epRankIdHccl);
        epWorldSizeHccl = epWorldSizeOriginal_;
    } else {
        statusFlagBaseGM = GetStatusDataSpaceGm(winContext_[COMM_EP_IDX]);
        epRankIdHccl = Mc2Kernel::GetRankId(winContext_[COMM_EP_IDX]);
        epWorldSizeHccl = Mc2Kernel::GetRankDim(winContext_[COMM_EP_IDX]);
    }
    statusBaseGM = statusFlagBaseGM;
    selfDataStatusGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)(statusFlagBaseGM + FLAG_FIELD_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_, moeExpertNum_, epWorldSizeOriginal_, globalBS_, dataStateBuf);
    // AscendC::printf("[kfm] dataState_ %d\n", dataState_);
    statusDataSpaceGM_ = GetWindStateAddrByRankId(epRankIdHccl);
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XInType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    
    uint32_t tokenWinSize = totalShmemSize_ - GetStageDoneRegionSize() - axisBS_ * hCommuSize_ -
        GetUrmaCtrlReserveSize() -
        FLAG_FIELD_OFFSET - aivNum_ * WIN_ADDR_ALIGN;
    shmemDataSizeOffset_ = dataState_ * (tokenWinSize / BUFFER_NUM);
                        //  + axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;
    winDataSizeOffset_ = dataState_ * (totalWinSize_ / BUFFER_NUM)
                         + axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::Init(GM_ADDR mc2Context, GM_ADDR x, 
    GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
    GM_ADDR tpSendCountsOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) // A3不支持MX量化，无需使能饱和模式
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    totalWinSize_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    if (tilingData->moeDistributeDispatchV2Info.isMc2Context) {
        isMc2Context_ = true;
        // Using Mc2Context instead of hccl context
        mc2Context_ = (__gm__ Mc2MoeContext*)mc2Context;
    } else {
        winContext_[COMM_EP_IDX] = (__gm__ HcclOpParam*)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        auto realWinSize = GetWinSize(winContext_[COMM_EP_IDX]);
        CheckWindowSize(totalWinSize_, realWinSize, tpipe_, expandXOut);
    }
    xGMTensor_.SetGlobalBuffer((__gm__ XInType*)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ uint8_t*)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t*)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(expandIdxOut));
    elasticInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(elasticInfo));
    scalesGMTensor_.SetGlobalBuffer((__gm__ float*)scales);
    shmemBuffer_ = tilingData->moeDistributeDispatchV2Info.shmemBuffer;
    SetTilingDataAndCal(tilingData);
    ClearStageDoneFlags();
    // InitHeadRecord();
    if (isPerformanceFlag_) {
        performanceInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(performanceInfo));
    }
    SetDataStatus();
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut;
    sendTpCountOutGM_ = tpSendCountsOut;
    // recvCntWorkspaceGM_ = workspaceGM;
    recvCntWorkspaceGM_ = GetWindStateAddrByRankId(epRankIdOriginal_) + rscvStatusNum_ * UB_ALIGN + 
        aivUsedCumSum_ * aivUsedCumSum_ * UB_ALIGN + aivUsedCumSum_ * aivNum_ * UB_ALIGN;
    statusSpaceGM_ = GetWindStateAddrByRankId(epRankIdOriginal_);
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float*)(statusSpaceGM_));
    selfRankWinInGMTensor_.SetGlobalBuffer((__gm__ float*)(statusDataSpaceGM_));
    windowGM_ = GetWindAddrByRankId(epRankIdOriginal_);
    // AscendC::printf("[kfm] windowGM_ %p\n", windowGM_);
    hCopyParams_ = {1U, static_cast<uint32_t>(copyInAxisH_ * sizeof(XInType)), 0U, 0U};
    dataStateParams_ = {1U, sizeof(uint32_t), 0U, 0U};
    expertIdsSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;

    // AscendC::printf("[state] rank %d aiv %d dataState %u windowGM %p statusGM %p\n",
    //     epRankId_, aivId_, dataState_, windowGM_, statusDataSpaceGM_);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::FillTriple(
    LocalTensor<XOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k)
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = epRankId_;        // 0:epRankId index
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex;   // 1:token index
    xOutTint32(tokenQuantAlign_ + 2) = k;            // 2:topK value index
    // hzy: 携带该 token 的完整 top-k expert id，后续 URMA 二次转发后用于重建真实 top-k index。
    for (uint32_t i = 0; i < axisK_; i++) {
        xOutTint32(tokenQuantAlign_ + UB_ALIGN_DATA_COUNT + i) = validExpertIdsTensor_(tokenIndex * axisK_ + i);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::TokenToExpertInQuant(
    GlobalTensor<XOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex,
    uint32_t fillExpertIdx, uint32_t quantExpertIdx)
{
    DataCopyPadParams copyPadParams{true, 0U, 0U, 0U};
    LocalTensor<XInType> xInTensor = inQueue.AllocTensor<XInType>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    LocalTensor<uint8_t> singleByteTok = xInTensor.template ReinterpretCast<uint8_t>();
    // 由于MX以及PERGROUP量化在计算scales时每次搬入256字节数据，所以在token搬入前需要对空间填0，避免引入脏数据
    if constexpr ((QuantMode == MX_QUANT) || (QuantMode == PERGROUP_DYNAMIC_QUANT)) {
        Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XInType));
    }
#endif
    SyncFunc<HardEvent::V_MTE2>();
    DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, copyPadParams);
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XInType>();
    if constexpr (QuantMode > UNQUANT) {
        quantInst_.QuantProcess(tempTensor_, xInTensor, quantExpertIdx, scalesCount_, scalesGMTensor_);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    else { // 非量化直转hifp8
        Cast(floatLocalTemp_, xInTensor, RoundMode::CAST_NONE, axisH_);
        Cast(tempTensor_, floatLocalTemp_, RoundMode::CAST_ROUND, axisH_);
    }
#endif
    FillTriple(tempTensor_, srcTokenIndex, fillExpertIdx);
    inQueue.FreeTensor<XInType>(xInTensor);
    SyncFunc<AscendC::HardEvent::S_V>();
    LocalTensor<int32_t> tempTensorInt32 = tempTensor_.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outTensorInt32 = outTensor_.template ReinterpretCast<int32_t>();
    PipeBarrier<PIPE_V>(); // QuantProcess中的Cast操作 -> Copy搬运
    // 64 = 256 / sizeof(int32_t) 一次操作字节数; 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[flagPadOffset_ / sizeof(int32_t)], tempTensorInt32, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15}); 
    // 64：偏移前一次拷贝的256字节； 56 = （480 - 256） / sizeof(int32_t); 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[flagPadOffset_ / sizeof(int32_t) + 64], tempTensorInt32[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(dstWinGMTensor, outTensor_[flagPadOffset_ / sizeof(XOutType)], axisHCommu_);
    flagPadOffset_ = hCommuSize_ - flagPadOffset_;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::TokenToExpert(
    GlobalTensor<XOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
    uint32_t srcTokenIndex, uint32_t toExpertIndex, uint32_t toRankId, uint32_t dstExpertId, uint32_t dstTokenPreCnt)
{
    DataCopyPadParams copyPadParams{false, 0U, 0U, 0U};
    LocalTensor<XInType> xInTensor = inQueue.AllocTensor<XInType>();
    if constexpr (!IsSmoothScaleExist) {
        DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, copyPadParams);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    else {
        DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};
        DataCopyPadParams padParams = {true, 0, 0, 0};
        auto tmp = scalesGMTensor_.ReinterpretCast<uint8_t>();
        DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * copyInAxisH_], hCopyParams_, copyPadParams);
        DataCopyPad(xInTensor[Align32(copyInAxisH_)].template ReinterpretCast<uint8_t>(),
            tmp[srcTokenIndex * scaleInBytes_], scaleInParams, padParams);
    }
#endif
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XInType>();
    FillTriple(xInTensor, srcTokenIndex, toExpertIndex);
    SyncFunc<AscendC::HardEvent::S_V>();
    LocalTensor<int32_t> xInTensorInt32 = xInTensor.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outTensorInt32 = outTensor_.template ReinterpretCast<int32_t>();
    // 64 = 256 / sizeof(int32_t) 一次操作字节数; 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[flagPadOffset_ / sizeof(int32_t)], xInTensorInt32,
        uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    // 64：偏移前一次拷贝的256字节； 56 = （480 - 256） / sizeof(int32_t); 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[flagPadOffset_ / sizeof(int32_t) + 64], xInTensorInt32[64],
        uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    // outTensorInt32(flagPadOffset_ / sizeof(int32_t)) = 854;
    // PipeBarrier<PIPE_ALL>();
    DataCopy(dstWinGMTensor, outTensor_[flagPadOffset_ / sizeof(XOutType)], axisHCommu_);
    flagPadOffset_ = hCommuSize_ - flagPadOffset_;
    inQueue.FreeTensor<XInType>(xInTensor);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
    uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;                // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t newAivId = 0;
    if (isFront) {
        newAivId = aivId_;
    } else if (aivId_ >= aivCumSumStart_) { // aiv中后面aivUsedCumSum_个核给cusum计算使用
        newAivId = aivId_ - aivCumSumStart_;
    } else if (aivId_ >= (moeUsedAivNum_ + sharedUsedAivNum_)){
        newAivId = aivId_ - (moeUsedAivNum_ + sharedUsedAivNum_);
    } else if(aivId_ >= moeUsedAivNum_){
        newAivId = aivId_ - moeUsedAivNum_;
    }
    startTokenId = sendTokenNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SendToSharedExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * BUFFER_NUM / sizeof(float));
    PipeBarrier<PIPE_V>();
    // 分核
    uint32_t startTokenId, endTokenId, sendTokenNum;
    uint32_t curSendCnt = activeMaskBsCnt_ * sharedExpertNum_; // 参数 validBsCnt_、sharedExpertNum_、sharedUsedAivNum_
    SplitToCore(curSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= curSendCnt) {return;}
    // 发送
    GlobalTensor<XOutType> dstWinGMTensor;
    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_;  // 计算目的共享专家卡在其所在共享专家组的id
    for (uint32_t virtualTokenIndex = startTokenId; virtualTokenIndex < endTokenId; ++virtualTokenIndex) {
        uint32_t sendTokenIndex = virtualTokenIndex % activeMaskBsCnt_;
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        int32_t toRankId = idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_;
        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
        }
        dstWinGMTensor.SetGlobalBuffer((__gm__ XOutType*)(GetWindAddrByRankId(toRankId) + expertPerSizeOnWin_ * \
            epRankId_ + sendTokenIndex * hCommuSize_));
        uint32_t srcTokenIndex = sendTokenIndex;
        if (isExpertMaskFlag_) {
            srcTokenIndex = validBsIndexTensor_.GetValue(sendTokenIndex);
        }
        if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
            uint32_t fillExpertIdx = axisK_ + toSharedExpertIndex;
            uint32_t quantExpertIdx = toSharedExpertIndex;
            TokenToExpertInQuant(dstWinGMTensor, inQueue, srcTokenIndex, fillExpertIdx, quantExpertIdx);
        } else {
            TokenToExpert(dstWinGMTensor, inQueue, srcTokenIndex, axisK_ + toSharedExpertIndex);
        }
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalExpertSendNum(TBuf<> outBuf, TBuf<> expertMaskBuf)
{
    uint64_t maskCnt = 0;
    uint32_t mask = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    uint32_t compareCount = Ceil(mask * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(int32_t);
    LocalTensor<uint8_t> expertMaskTensorU8 = expertMaskBuf.Get<uint8_t>();
    LocalTensor<uint32_t> expertMaskTensorU32 = expertMaskBuf.Get<uint32_t>();
    LocalTensor<int32_t> gatherTempTensor = outBuf.Get<int32_t>();
    for (int32_t expertIndex = 0; expertIndex < sendNum_; expertIndex++) {
        int32_t dstExpertId = expertIndex + startId_;
        if ((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0)) {
            dstExpertId = delLastExpertId_;
        }
        CompareScalar(expertMaskTensorU8[maskSizePerExpert_ * expertIndex], validExpertIdsTensor_, dstExpertId, CMPMODE::EQ, compareCount);
        PipeBarrier<PIPE_V>();
        GatherMask(gatherTempTensor, validExpertIdsTensor_, expertMaskTensorU32[maskSizePerExpert_ * expertIndex / sizeof(uint32_t)],
            true, mask, {1, 1, 0, 0}, maskCnt); // 是否可以简化计算
        SyncFunc<AscendC::HardEvent::V_S>();
        tokenNumToExpertTensor_.SetValue(expertIndex, static_cast<uint32_t>(maskCnt));
    }
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * BUFFER_NUM / sizeof(float));
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SplitExpertNumToCore(uint32_t &delCurExpertGroupNum, uint32_t &groupIdx)
{
    sendNum_ = moeExpertNum_ / moeUsedAivNum_;
    remainderExpertNum_ = moeExpertNum_ % moeUsedAivNum_;
    startId_ = sendNum_ * aivId_;
    if (remainderExpertNum_ != 0) {
        int32_t remainderGroupSize = remainderExpertNum_;
        delLastExpertId_ = aivId_ % remainderExpertNum_;
        delCurExpertGroupNum = moeUsedAivNum_ / remainderGroupSize;
        if (delLastExpertId_ < moeUsedAivNum_ % remainderGroupSize) {
            delCurExpertGroupNum++;
        }
        groupIdx = aivId_ / remainderGroupSize;
        delLastExpertId_ += moeUsedAivNum_ * sendNum_;
        sendNum_ += 1;
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue,
    TBuf<> expertMaskBuf, TBuf<> outBuf)
{
    // 分核
    uint32_t delCurExpertGroupNum, groupIdx, calExpertIdsIdx;
    SplitExpertNumToCore(delCurExpertGroupNum, groupIdx);
    // 计算专家发送数据量 && 发送
    CalExpertSendNum(outBuf, expertMaskBuf);
    uint32_t maskN64Num = Ceil(expertIdsCnt_, 64); // 64：ScalarGetSFFValue按照64长度一次计算
    GlobalTensor<XOutType> dstWinGMTensor;
    LocalTensor<uint64_t> expertMaskTensorU64 = expertMaskBuf.Get<uint64_t>();
    for (int32_t index = 0; index < sendNum_; index++) {
        int32_t dstTokenPreCnt = 0;
        int32_t expertIndex = (index + epRankId_ % sendNum_) % sendNum_;
        int32_t maskExpertU64Cnt = maskSizePerExpert_ * expertIndex / sizeof(uint64_t);
        if (tokenNumToExpertTensor_(expertIndex) == 0) {
            continue;
        }
        int32_t dstExpertId = expertIndex + startId_;
        dstExpertId = ((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0)) ? delLastExpertId_ : dstExpertId;
        for (int32_t maskIndex = 0; maskIndex < maskN64Num; maskIndex++) {
            uint64_t dstExpInfoMask = expertMaskTensorU64(maskIndex + maskExpertU64Cnt);
            int64_t curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            while (curValidIdx >= 0) {
                calExpertIdsIdx = curValidIdx + SFFVALUE_SIZE * maskIndex; // 64：ScalarGetSFFValue按照64长度一次计算
                if (calExpertIdsIdx >= expertIdsCnt_) {
                    break;
                }
                int32_t topKIndex = calExpertIdsIdx % axisK_;
                int32_t srcTokenIndex = calExpertIdsIdx / axisK_;
                int32_t toRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
                if (isScalingDownFlag_) {
                    toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
                }
                GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindAddrByRankId(toRankId) +
                                                (expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
                                                hCommuSize_ * dstTokenPreCnt); // 计算地址偏移
                dstWinGMTensor.SetGlobalBuffer((__gm__ XOutType*)rankGM);
                if (!((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0) && (dstTokenPreCnt % delCurExpertGroupNum != groupIdx))) {
                    if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
                        uint32_t quantExpertIdx = dstExpertId + sharedExpertNum_;
                        TokenToExpertInQuant(dstWinGMTensor, inQueue, srcTokenIndex, topKIndex, quantExpertIdx);
                    } else {
                        TokenToExpert(dstWinGMTensor, inQueue, srcTokenIndex, topKIndex);
                    }
                }
                dstTokenPreCnt++;
                uint64_t cleanMask = ~(uint64_t(1) << curValidIdx);
                dstExpInfoMask = cleanMask & dstExpInfoMask; // 将当前64bit中处理的1置0
                curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            }
        }
    }
}


template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SendToMoeExpertNew(TQue<QuePosition::VECIN, 1> inQueue,
    TBuf<> expertMaskBuf, TBuf<> outBuf)
{
    // 按Token分核
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * BUFFER_NUM / sizeof(float));
    PipeBarrier<PIPE_V>();
    uint32_t activeMoeStageNum = GetActiveMoeStageNum();
    if (aivId_ >= activeMoeStageNum) {
        return;
    }
    sendNum_ = axisBS_ / activeMoeStageNum;
    startId_ = sendNum_ * aivId_;
    uint32_t reaminTokenNum = axisBS_ - sendNum_ * activeMoeStageNum;
    if(aivId_ < reaminTokenNum){
        sendNum_ += 1;
        startId_ += aivId_;
    } else {
        startId_ += reaminTokenNum;
    }
    if(sendNum_ == 0)
        return;
    
    GlobalTensor<XOutType> sendBufferTensor;
    GlobalTensor<XOutType> dstWinGMTensor;

    for (int32_t index = 0; index < sendNum_; index++) {
        int32_t tokenId = startId_ + index;
        sendBufferTensor.SetGlobalBuffer((__gm__ XOutType*)GetSendBufferAddrByTokenId(tokenId));
        // PipeBarrier<PIPE_ALL>(); 
        if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
            uint32_t quantExpertIdx = 0 + sharedExpertNum_;
            TokenToExpertInQuant(sendBufferTensor, inQueue, tokenId, 0, quantExpertIdx);
        } else {
            TokenToExpert(sendBufferTensor, inQueue, tokenId, 0);
        }
        PipeBarrier<PIPE_MTE3>();
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        PipeBarrier<PIPE_ALL>();

        // GlobalTensor<float> sendBufferTensorFloat = sendBufferTensor.template ReinterpretCast<float>();
        // for (uint32_t i = 0; i < blockCntPerToken_; ++i) {
        //     float iFlag = sendBufferTensorFloat.GetValue((i * SPLIT_BLOCK_SIZE + SPLIT_BLOCK_DATA_SIZE) / sizeof(float));
        //     AscendC::printf("[kfm] checkSendBufferTokenFlag aivId_ %d tokenId %d i %d flag %f\n", aivId_, tokenId, i, iFlag);
        // }

        // GlobalTensor<int32_t> sendBufferTensorInt32 = sendBufferTensor.template ReinterpretCast<int32_t>();
        // AscendC::printf("[kfm] aivId_ %d tokenId %d SendBufferAddrByTokenId %p x(0) %d x(480 byte) %f triple %d %d %d\n", 
        //     aivId_, tokenId, GetSendBufferAddrByTokenId(tokenId), sendBufferTensorInt32.GetValue(0),
        //     sendBufferTensorFloat.GetValue(480 / sizeof(float)),
        //     sendBufferTensorInt32.GetValue(15264 / sizeof(int32_t)),
        //     sendBufferTensorInt32.GetValue(15268 / sizeof(int32_t)),
        //     sendBufferTensorInt32.GetValue(15272 / sizeof(int32_t)));  
    }
    PipeBarrier<PIPE_ALL>();
    DebugClock(3);
    

    // PipeBarrier<PIPE_ALL>();
    // printf("sendInfo-debug: srcRank %d, aivid %d sendNum %d tokenbegin%d\n",epRankId_, aivId_, sendNum_, startId_);
    // for (int32_t i = 0; i < sendNum_; i++) {
    //     int32_t tokenId = startId_ + i;
    //     sendBufferTensor.SetGlobalBuffer((__gm__ XOutType*)GetSendBufferAddrByTokenId(tokenId));
    //     // PipeBarrier<PIPE_ALL>();
    //     DataCopy(outTensor_, sendBufferTensor, axisHCommu_);
    //     PipeBarrier<PIPE_ALL>();
    //     for (int32_t j = 0; j < axisK_ ; j++) {
    //         int32_t expertIdx = tokenId * axisK_ + j;
    //         int32_t expertId = validExpertIdsTensor_(expertIdx);
    //         uint32_t dstTokenIdx = 0;
    //         for(int32_t k = 0; k < expertIdx; k++){
    //             if(validExpertIdsTensor_(k) == expertId)
    //                 dstTokenIdx++;
    //         }
    //         dstWinGMTensor.SetGlobalBuffer((__gm__ XOutType*)(GetWindAddrByRankId(expertId / moeExpertNumPerRank_) 
    //             + expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ + expertId % moeExpertNumPerRank_) + dstTokenIdx * hCommuSize_));
    //         PipeBarrier<PIPE_ALL>();
    //         // printf("sendInfo: srcRank %d, tokenId %d k %d dst rank %d dstExpertId %d dstidx %d dstaddr %p\n",epRankId_, tokenId, j, expertId / moeExpertNumPerRank_, expertId, dstTokenIdx, dstWinGMTensor.GetPhyAddr());

    //         // printf("sendInfo: srcRank %d, aivid %d,  tokenId %d dstaddr %p\n",epRankId_, aivId_, tokenId, dstWinGMTensor.GetPhyAddr());
    //         DataCopy(dstWinGMTensor, outTensor_, axisHCommu_);
    //         PipeBarrier<PIPE_ALL>();
    //     }
    // }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::AllToAllDispatch()
{
    // 使用的全局参数
    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<> tempBuf, outBuf, expertIdsBuf, expertMaskBuf;
    TBuf<> smoothScalesBuf, tokenNumToExpertBuf, receiveDataCastFloatBuf;
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(inQueue, BUFFER_NUM, hSizeAlignBlock_);
    tpipe_->InitBuffer(outBuf, hCommuSize_ * BUFFER_NUM);
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    outTensor_ = outBuf.Get<XOutType>();
    bool needMaskCalFlag = (isTokenMaskFlag_ || isExpertMaskFlag_ || zeroComputeExpertNum_ != 0);
    if (needMaskCalFlag) {
        tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);
    }
    if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
        hOutAlignUbSize_ = Ceil(hOutSizeAlignBlock_, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(tempBuf, hOutAlignUbSize_);
        tpipe_->InitBuffer(receiveDataCastFloatBuf, maxSize_);
        tpipe_->InitBuffer(smoothScalesBuf, maxSize_);
        tempTensor_ = tempBuf.Get<XOutType>();
        floatLocalTemp_ = receiveDataCastFloatBuf.Get<float>();
        smoothScalesTensor_ = smoothScalesBuf.Get<float>();
        if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            floatLocalAbsTemp_ = smoothScalesBuf.Get<float>();
        }
        dstExpBuf_ = receiveDataCastFloatBuf; // 内存复用
        subExpBuf_ = smoothScalesBuf;         // 内存复用
    } else if (needMaskCalFlag) {
        tpipe_->InitBuffer(dstExpBuf_, maxSize_);
        tpipe_->InitBuffer(subExpBuf_, maxSize_);
    }
    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf, dynamicScalesOutGMTensor_);
    PipeBarrier<PIPE_ALL>();
    DebugClock(1);
    ExpIdsCopyAndMaskCal();
    PipeBarrier<PIPE_ALL>();
    DebugClock(2);
    if (activeMaskBsCnt_ == 0) {
        return;
    }

    if ((aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0)) {
        SendToSharedExpert(inQueue, outBuf);
    } else {
        maskSizePerExpert_ = Ceil((expertIdsBufSize_ / sizeof(int32_t)) / 8, UB_ALIGN) * UB_ALIGN; // 8 is 1byte->8bit
        uint32_t expertMaskBufSize = maskSizePerExpert_ * Ceil(moeExpertNum_, aivNum_);
        tpipe_->InitBuffer(expertMaskBuf, expertMaskBufSize);
        tpipe_->InitBuffer(tokenNumToExpertBuf, Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
        tokenNumToExpertTensor_ = tokenNumToExpertBuf.Get<int32_t>();
        LocalTensor<int32_t> expertMaskTensor = expertMaskBuf.Get<int32_t>();
        Duplicate<int32_t>(expertMaskTensor, 0, int32_t(expertMaskBufSize / sizeof(int32_t)));
        SyncFunc<AscendC::HardEvent::V_S>();
        // hzy: Moe token 先写本地 send buffer，URMA 核后续按 expert/rank 顺序二次转发。
        SendToMoeExpertNew(inQueue, expertMaskBuf, outBuf);
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt)
{
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> subExpIdTensor = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(dstExpIdTensor, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor, validExpertIdsTensor_, dstExpIdTensor, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor, dstExpIdTensor, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalAndSendCnt()
{
    TBuf<> checkBuf;
    tpipe_->InitBuffer(checkBuf, UB_ALIGN);
    LocalTensor<int32_t> checkTensor = checkBuf.Get<int32_t>();

    uint32_t startExpertId, endExpertId, sendExpertNum;
    uint32_t maskCnt = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    SplitToCore(totalExpertNum_, aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum, false);
    if (startExpertId >= totalExpertNum_) {return;}
    uint64_t mask[2] = { 0x101010101010101, 0 }; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_ / 8, 1, 8); // 0x3F800000为float的1 8为一次操作8个block
    PipeBarrier<PIPE_ALL>();

    SyncFunc<AscendC::HardEvent::V_S>();

    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t curExpertId = startExpertId; curExpertId < endExpertId; ++curExpertId) {
        int32_t curExpertCnt = 0;
        int32_t cntPosIndex = (curExpertId - startExpertId) * 8 + 1;               // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
        if ((curExpertId < sharedExpertRankNum_) && (activeMaskBsCnt_ > 0)) {     // 当前处理专家id为共享专家
            if (curExpertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
                curExpertCnt = activeMaskBsCnt_;
            }
        } else if (sendToMoeExpTokenCnt_ > 0) { // 当前处理卡为moe专家卡
            int32_t curMoeExpertId = curExpertId - sharedExpertRankNum_;
            CalTokenSendExpertCnt(curMoeExpertId, maskCnt, curExpertCnt);
        }
        statusTensor_.SetValue(cntPosIndex, curExpertCnt);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t dstRankId = curExpertId;
        uint32_t offset = STATE_OFFSET * epRankId_;
        if (curExpertId >= sharedExpertRankNum_) {
            dstRankId = ((curExpertId - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset += ((curExpertId - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_ * STATE_OFFSET);
        }
        if (isScalingDownFlag_) {
            dstRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId);
        }
        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindStateAddrByRankId(dstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t*)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(curExpertId - startExpertId) * UB_ALIGN_DATA_COUNT], UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        // SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

        // DataCopy<int32_t>(checkTensor, rankGMTensor, UB_ALIGN_DATA_COUNT);
        // SyncFunc<AscendC::HardEvent::MTE2_S>();

        // AscendC::printf("[sendCntAfter] rank %d curExpert %u dstRank %u offset %u flag %f cnt %d raw0 0x%x raw1 %d addr %p\n",
        //     epRankId_, curExpertId, dstRankId, offset,
        //     checkTensor.ReinterpretCast<float>()(0),
        //     checkTensor(1),
        //     checkTensor(0),
        //     checkTensor(1),
        //     rankGM);


    }
    // reset操作前需确保前面操作完成
    PipeBarrier<PIPE_ALL>();
}
template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::BufferInit()
{
    uint32_t waitStatusBufSize = Ceil((recStatusNumPerCore_ * UB_ALIGN), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = epWorldSize_ * moeExpertNumPerRank_ * sizeof(float);
    uint64_t gatherMaskOutSize = (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    uint64_t sumContinueAlignSize = Ceil((aivNum_ * sizeof(float)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);           // recStatusNumPerCore_32对齐后大小  * 32B
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);                // 48 * 32B
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN);               // 48 * 32B
    tpipe_->InitBuffer(sumContinueBuf_, sumContinueAlignSize);          // 48 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);                       // 96 B
    if (isPerformanceFlag_) {
        uint32_t performanceFlagSize = recStatusNumPerCore_ * sizeof(int32_t);
        uint32_t performanceFlagSizeAlign = Ceil(performanceFlagSize, UB_ALIGN) * UB_ALIGN;
        uint32_t performanceInfoSize = epWorldSizeOriginal_ * sizeof(int64_t);
        uint32_t performanceInfoSizeAlign = Ceil(performanceInfoSize, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(performanceInfoBuf_, performanceInfoSizeAlign);
        performanceInfoTensor_ = performanceInfoBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceInfoTensor_, 0, performanceInfoSizeAlign / sizeof(int32_t));
        tpipe_->InitBuffer(performanceFlagBuf_, performanceFlagSizeAlign);
        performanceFlagTensor_ = performanceFlagBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceFlagTensor_, 0, performanceFlagSizeAlign / sizeof(int32_t));
        if (isScalingDownFlag_) {
            uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t);
            uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
            tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
            elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
            DataCopyExtParams elasticInfoParams = {1U, static_cast<uint32_t>(
                (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
            DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor_, elasticInfoParams, elasticInfoCopyPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
        }
    }
    uint32_t statusBufSize = rscvStatusNum_ * UB_ALIGN;
    uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t workLocalBufSize = Ceil(epWorldSize_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
    tpipe_->InitBuffer(workLocalBuf_, workLocalBufSize);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::WaitDispatchClearStatus()
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0}; // 一次操作256字节，每8个数将首个设置为0
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8); // 8 = 256 / 32
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::GatherSumRecvCnt(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
    LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, 2);  // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;               // 源操作数每个datablock只需要处理两个元素
    SyncFunc<AscendC::HardEvent::S_V>();

    // 将当前核对应的专家 recvCnt 收集到gatherMaskOutTensor
    uint64_t recvCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
        {1, (uint16_t)recStatusNumPerCore_, 1, 0}, recvCnt);
    PipeBarrier<PIPE_V>();

    // 对当前核对应的专家recv cnt求和
    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) // 对inner要求32对齐
        * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);

    // 把当前核的所有专家recv cnt之和写到状态区
    uint32_t newAivId = aivId_ - aivCumSumStart_;
    // 每个核把sumOfRecvCnt重复写 aivUsedCumSum_ 份
    LocalTensor<float> sumCoreFP32Tensor = sumCoreBuf_.Get<float>();
    uint64_t maskArrayCount[2] = {0x0101010101010101, 0};
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8); // 8 = 256 / 32
    // 每次处理256字节，8个datablock，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFP32Tensor, sumOfRecvCnt, maskArrayCount, repeatTimes, 1, 8);
    uint64_t maskArrayFlag[2] = {0x0202020202020202, 0};
    Duplicate<float>(sumCoreFP32Tensor, static_cast<float>(1.0), maskArrayFlag, repeatTimes, 1, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(rscvStatusNum_ * STATE_OFFSET + newAivId * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)], sumCoreFP32Tensor, sumIntriParams);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId)
{
    outLocal = gatherMaskOutBuf_.Get<int32_t>();
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, static_cast<uint16_t>(aivUsedCumSum_ - 1), 0};
    LocalTensor<float> sumLocalTensor = sumLocalBuf_.Get<float>();
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> sumContinueTensor = sumContinueBuf_.Get<float>();
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);

    uint32_t mask = 2;
    uint64_t recvCnt = 0;
    uint32_t innerSumParams = Ceil(aivUsedCumSum_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, innerSumParams, aivUsedCumSum_};
    int32_t cumSumFlag = 0;
    gatherSumPattern.SetValue(0, 2);
    SyncFunc<AscendC::HardEvent::S_V>();

    // 获取状态区中每个核的recvCnt
    while (true) {
        DataCopy(sumLocalTensor, selfRankWinInGMTensor_[(rscvStatusNum_ * STATE_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumIntriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(aivUsedCumSum_), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        Sum(recvCntSumOutTensor, sumContinueTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = static_cast<int32_t>(recvCntSumOutTensor.GetValue(0));
        if (cumSumFlag == aivUsedCumSum_) {
            break;
        }
    }

    // 0核前面所有核recv cnt总和是0
    if (newAivId == 0) {
        outLocal.SetValue(0, 0);
    } else {
        mask = 1;
        recvCnt = 0;
        gatherSumPattern.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(newAivId), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }
    // 清除 flag 用于下次aivUsedCumSum_软同步
    LocalTensor<float> sumCoreFp32Tensor = sumLocalBuf_.Get<float>();
    // 一次处理256字节，8个datablock
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8);
    // 64 = 256 / sizeof(float) 一次操作字节数，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFp32Tensor, static_cast<float>(0), 64, repeatTimes, 1, 8);
    DataCopyParams cleanParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(rscvStatusNum_ * STATE_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumCoreFp32Tensor, cleanParams);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::WaitDispatch()
{
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = 1;
    gatherTmpTensor.SetValue(0, 1);
    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    uint64_t performanceTimeStart = static_cast<uint64_t>(GetSystemCycle());

    uint32_t spin = 0;

    while (sumOfFlag != compareTarget) {
        for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                windowInstatusFp32Tensor_[(startStatusIndex_ + i) * FLAG_OFFSET]);
        }
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)], intriParams);
        if (isPerformanceFlag_) {
            RecordRankCommDuration(performanceInfoTensor_, performanceTimeStart);
 	    }
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);

        auto statusI32 = statusFp32Tensor_.ReinterpretCast<int32_t>();
        // if ((spin++ & 0xffff) == 0) {
        //     AscendC::printf(
        //         "[wd] rank %d sum %f target %f "
        //         "f0 %f c0 %d f1 %f c1 %d f2 %f c2 %d f3 %f c3 %d\n",
        //         epRankId_, sumOfFlag, compareTarget,
        //         statusFp32Tensor_(0 * UB_ALIGN_DATA_COUNT), statusI32(0 * UB_ALIGN_DATA_COUNT + 1),
        //         statusFp32Tensor_(1 * UB_ALIGN_DATA_COUNT), statusI32(1 * UB_ALIGN_DATA_COUNT + 1),
        //         statusFp32Tensor_(2 * UB_ALIGN_DATA_COUNT), statusI32(2 * UB_ALIGN_DATA_COUNT + 1),
        //         statusFp32Tensor_(3 * UB_ALIGN_DATA_COUNT), statusI32(3 * UB_ALIGN_DATA_COUNT + 1));
        // }

    }
    // RunPosRecord(RUNPOS_CALCUMSUM); // 维测打点
    if (isPerformanceFlag_) {
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        SetAtomicMax<int32_t>();
        DataCopyExtParams performanceInfoCopyParams{1U, static_cast<uint32_t>(epWorldSizeOriginal_* sizeof(int64_t)),
            0U, 0U, 0U};
        DataCopyPad(performanceInfoGMTensor_, performanceInfoTensor_, performanceInfoCopyParams);
        SetAtomicNone();
    }
    // 清状态
    WaitDispatchClearStatus();
    GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalRecvAndSetFlag()
{
    // check flag 用于 aivUsedCumSum_ 软同步并计算 aivUsedCumSum_ 个核各自的recvCount
    LocalTensor<int32_t> outCountLocal;
    uint32_t newAivId = aivId_ - aivCumSumStart_;
    GetCumSum(outCountLocal, newAivId);
    // 计算epRecvCnt
    uint32_t preSum = outCountLocal.GetValue(0);
    uint32_t curCnt = preSum;
    statusTensor_ = waitStatusBuf_.Get<int32_t>();
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        uint32_t i = index - startStatusIndex_;
        uint32_t count = statusTensor_.GetValue(i * UB_ALIGN_DATA_COUNT + 1);
        curCnt += count;
        outCountLocal.SetValue(i, curCnt);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    GM_ADDR wAddr = (__gm__ uint8_t*)(recvCntWorkspaceGM_);
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer((__gm__ int32_t*)wAddr);
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(recStatusNumPerCore_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(sendCountsGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);
    // 复制aivNum_份
    for (uint32_t index = 0; index < aivNum_; index++) {
        DataCopyPad(workspaceGlobal[index * rscvStatusNum_ + startStatusIndex_], outCountLocal, dataCopyOutParams);
    }
    uint8_t repeatTimes = Ceil(aivNum_, 8);  // 一次处理256字节，8个datablock
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivNum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    LocalTensor<int32_t> syncOnCoreTensor = sumCoreBuf_.Get<int32_t>();
    LocalTensor<float> syncOnCoreFP32Tensor = sumCoreBuf_.Get<float>();
    // 每次处理256字节，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<int32_t>(syncOnCoreTensor, static_cast<int32_t>(1), SIZE_ALIGN_256 / sizeof(int32_t), repeatTimes, 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(rscvStatusNum_ * STATE_OFFSET + aivUsedCumSum_ * aivUsedCumSum_ * UB_ALIGN + newAivId * UB_ALIGN) / sizeof(float)], syncOnCoreFP32Tensor, sumIntriParams);  // 软同步
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SetExpertTokenNums()
{
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    DataCopyParams totalStatusCopyParams{static_cast<uint16_t>(localExpertNum * epWorldSize_), 1, 0, 0};
    LocalTensor<float> totalStatusTensorFp32 = statusBuf_.Get<float>();
    DataCopy(totalStatusTensorFp32, windowInstatusFp32Tensor_, totalStatusCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    int64_t expertTokenNumCumsum = 0;
    LocalTensor<int64_t> expertTokenNumsLocalTensor = tokenNumBuf_.Get<int64_t>();
    LocalTensor<float> expertTokenNumTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), 0);
    LocalTensor<float> workLocalTensor = workLocalBuf_.Get<float>();

    for (uint32_t localExpertIdx = 0; localExpertIdx < localExpertNum; ++localExpertIdx) {
        LocalTensor<float> expertStatusTensor = statusBuf_.GetWithOffset<float>(
            epWorldSize_ * UB_ALIGN / static_cast<uint32_t>(sizeof(float)), localExpertIdx * epWorldSize_ * UB_ALIGN);
        uint32_t mask = 2;
        SyncFunc<AscendC::HardEvent::S_V>();
        ReduceSum(expertTokenNumTensor, expertStatusTensor, workLocalTensor, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        int64_t expertTokenNum = static_cast<int64_t>(expertTokenNumTensor.ReinterpretCast<int32_t>().GetValue(0));
        expertTokenNumCumsum += expertTokenNum;
        if (expertTokenNumsType_ == 0) {
            expertTokenNumsLocalTensor.SetValue(localExpertIdx, expertTokenNumCumsum);
        } else {
            expertTokenNumsLocalTensor.SetValue(localExpertIdx, expertTokenNum);
        }
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams expertTokenNumsCopyParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)),
                                                0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor, expertTokenNumsCopyParams);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::RecordRankCommDuration(
    LocalTensor<int32_t> &performanceInfoTensor, uint64_t startTime)
{
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint64_t endTime = static_cast<uint64_t>(GetSystemCycle());
    int32_t duration = static_cast<int32_t>((endTime - startTime) / CYCLES_PER_US);
    for (uint32_t i = 0; i < recStatusNumPerCore_; i ++) {
        float statusFp32 = statusFp32Tensor_.GetValue(i * FLAG_OFFSET);
        int32_t performanceFlag = performanceFlagTensor_.GetValue(i);
        if (statusFp32 > float(0.5) && performanceFlag == 0) {
            performanceFlagTensor_.SetValue(i, 1);
            uint32_t fromLocalRankId = (startStatusIndex_ + i) % epWorldSize_;
            uint32_t fromRankId = isScalingDownFlag_ ?
                elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + fromLocalRankId) :
                fromLocalRankId;
            int32_t savedTime = performanceInfoTensor.GetValue(fromRankId * DURATION_OFFSET);
            int32_t newValue = (duration > savedTime) ? duration : savedTime;
            if (newValue != savedTime) {
                 // 乘2 使用int32_t是因为atomicAdd不支持int64_t类型，这里只赋值到int64_t的低32位。
                performanceInfoTensor.SetValue(fromRankId * DURATION_OFFSET, duration);
            }
        }
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalCumSum()
{
    // 进来的核统一做发送，各专家的token总数发送
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(dstExpBuf_, maxSize_);           // BS * K * 4
    tpipe_->InitBuffer(subExpBuf_, maxSize_);           // BS * K * 4
    tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);      // BS * K * 4
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    tpipe_->InitBuffer(statusBuf_, statusCntAlign_ * UB_ALIGN);
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
    statusTensor_ = statusBuf_.Get<int32_t>();
    ExpIdsCopyAndMaskCal();
    CalAndSendCnt();
    DebugClock(2);
    SplitToCore(rscvStatusNum_, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_, false);
    tpipe_->Reset();
    BufferInit();
    // AscendC::printf("[cum] rank %d aiv %d before WaitDispatch start %u rec %u\n",
    //     epRankId_, aivId_, startStatusIndex_, recStatusNumPerCore_);
    WaitDispatch();
    DebugClock(3);
    // AscendC::printf("[cum] rank %d aiv %d after WaitDispatch\n", epRankId_, aivId_);
    CalRecvAndSetFlag();
    DebugClock(4);
    // AscendC::printf("[cum] rank %d aiv %d after CalRecvAndSetFlag\n", epRankId_, aivId_);
    // 使用newAivId为0的核进行计算
    if (aivId_ == aivCumSumStart_) {
        SetExpertTokenNums();
    }
    DebugClock(5);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::WaitCumSumFlag()
{
    // Check cumsum is finished
    int32_t cumSumFlag = 0;
    int32_t targetFlag = aivUsedCumSum_ * UB_ALIGN_DATA_COUNT;
    uint32_t cumSumFlagOffset = (rscvStatusNum_ * STATE_OFFSET + aivUsedCumSum_ * aivUsedCumSum_ * UB_ALIGN + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float);
    uint32_t innerSumParams = aivUsedCumSum_ * UB_ALIGN / sizeof(float);
    SumParams sumFlagParams{1, innerSumParams, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT};
    LocalTensor<float> statusSumOutTensor = scalarBuf_.Get<float>();

    for (uint32_t i = 0; i < aivUsedCumSum_; ++i) {
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            selfRankWinInGMTensor_[cumSumFlagOffset + i * UB_ALIGN_DATA_COUNT]);
    }

    while (true) {
        DataCopy(statusFp32Tensor_, selfRankWinInGMTensor_[cumSumFlagOffset], aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);
        if (cumSumFlag == targetFlag) {
            break;
        }
    }
    // RunPosRecord(RUNPOS_CUMSUMFLAG); // 维测打点
    // Clean flag for next round
    Duplicate<float>(statusCleanFp32Tensor_, static_cast<float>(0), aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[cumSumFlagOffset], statusCleanFp32Tensor_, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum)
{
    // 获取cumSum
    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(recvCntWorkspaceGM_));
    DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    for (uint32_t index = startId_; index < endId_; index++) { // 从sendCnt中挑选当前有发送过来的卡的token数量
        expertMapTensor_(validNum) = index;
        if (index == 0) {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index);
        } else {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index) - sendCntTensor_(index - 1);
        }
        if (expertLeftNumTensor_(validNum) != 0) {
            // AscendC::printf("[kfm] expertLeftNumTensor_ aivId_ %d validNum-1 %d, expertLeftNumTensor_(validNum-1) %d\n", aivId_, validNum, expertLeftNumTensor_(validNum));
            validNum += 1;
        } 
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CheckDataArriveWithFlag(uint32_t srcExpDataIdx,
    int32_t beginIdx, int32_t copyCnt)
{
    uint64_t rsvdCnt = 0;
    uint32_t arriveFlagNum = 0;
    uint32_t flagNum = blockCntPerToken_ * uint32_t(copyCnt);
    uint32_t compareCount = Ceil(flagNum, COMPARE_COUNT_PER_BLOCK) * COMPARE_COUNT_PER_BLOCK;
    uint32_t compResultU64Num = Ceil(flagNum, 64); // 64：按照64bit位进行划分
    DataCopyExtParams expFlagCopyParams{static_cast<uint16_t>(flagNum), static_cast<uint32_t>(sizeof(float)),
        static_cast<uint32_t>(SPLIT_BLOCK_SIZE - sizeof(float)), 0, 0};

    DataCopyExtParams oneTokenFlagCopyParams{
        static_cast<uint16_t>(blockCntPerToken_),
        static_cast<uint32_t>(UB_ALIGN),
        static_cast<uint32_t>(SPLIT_BLOCK_SIZE - UB_ALIGN),
        0U,
        0U
    };

    const uint32_t tokenStride = hCommuSize_ / sizeof(float);

    DataCopyPadExtParams<float> expFlagPadParams{false, 0U, 0U, 0U};
    GlobalTensor<float> dataFlagGlobal;
    GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcExpDataIdx * expertPerSizeOnWin_ + // 拿到第一个起始位置
        beginIdx * hCommuSize_ + SPLIT_BLOCK_DATA_SIZE;
    // AscendC::printf("[kfm] aivId %d srcExpDataIdx %d beginIdx %d wAddr %p\n", aivId_, srcExpDataIdx, beginIdx, wAddr);
    dataFlagGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));
    // AscendC::printf("[kfm121] aivId %d dataFlagGlobal %f\n", aivId_, dataFlagGlobal.GetValue(0));

    // for (uint32_t i = 0; i < flagNum; ++i) {
    //     AscendC::DataCacheCleanAndInvalid<float,
    //         AscendC::CacheLine::SINGLE_CACHE_LINE,
    //         AscendC::DcciDst::CACHELINE_ALL>(dataFlagGlobal[i * SPLIT_BLOCK_COUNT]);
    // }

    // SyncFunc<AscendC::HardEvent::S_MTE2>();

    // uint32_t localExpertNum =
    //     isShareExpertRankFlag_ ? 1U :
    //     moeExpertNumPerRank_;

    // uint32_t maxDataBlock = epWorldSize_ * localExpertNum;

    // bool invalid =
    //     srcExpDataIdx >= maxDataBlock ||
    //     beginIdx < 0 ||
    //     copyCnt <= 0 ||
    //     static_cast<uint32_t>(beginIdx) >= axisMaxBS_ ||
    //     static_cast<uint32_t>(beginIdx) +
    //         static_cast<uint32_t>(copyCnt) > axisMaxBS_;

    // if (invalid) {
    //     AscendC::printf(
    //         "[FLAG_RANGE_ERROR] rank %d aiv %d "
    //         "srcBlock %u maxBlock %u begin %d copy %d "
    //         "axisMaxBS %u hCommu %u\n",
    //         epRankId_,
    //         aivId_,
    //         srcExpDataIdx,
    //         maxDataBlock,
    //         beginIdx,
    //         copyCnt,
    //         axisMaxBS_,
    //         hCommuSize_);
    //     return 0;
    // }

    // for (uint32_t token = 0; token < static_cast<uint32_t>(copyCnt); ++token) {
    //     DataCopyPad(
    //         flagRecvTensor_[token * blockCntPerToken_ * UB_ALIGN_DATA_COUNT],
    //         dataFlagGlobal[token * tokenStride],
    //         oneTokenFlagCopyParams,
    //         expFlagPadParams);
    // }
    // return uint32_t(copyCnt);

    // DataCopyPad(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams, expFlagPadParams);

    // DataCopyParams flagCopyParams{
    //     static_cast<uint16_t>(flagNum),
    //     1U,
    //     static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE / UB_ALIGN),
    //     0U
    // };
    // DataCopy(flagRecvTensor_, dataFlagGlobal, flagCopyParams);

    DataCopyParams flagCopyParams{
        static_cast<uint16_t>(blockCntPerToken_),
        1U,
        static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE / UB_ALIGN),
        0U
    };
    for (uint32_t token = 0; token < static_cast<uint32_t>(copyCnt); ++token) {
        DataCopy(
            flagRecvTensor_[token * blockCntPerToken_ * UB_ALIGN_DATA_COUNT],
            dataFlagGlobal[token * tokenStride],
            flagCopyParams);
    }

    SyncFunc<AscendC::HardEvent::MTE2_V>();
    // AscendC::printf("[kfm122] aivId %d\n", aivId_);
    GatherMask(flagGatherOutTensor_, flagRecvTensor_, flagRecvGatherMask_, true, uint32_t(1),
         {1, (uint16_t)(flagNum), 1, 0}, rsvdCnt); 
    PipeBarrier<PIPE_V>();
    CompareScalar(flagCompResultU8_, flagGatherOutTensor_, float(1), AscendC::CMPMODE::EQ, compareCount);
    SyncFunc<AscendC::HardEvent::V_S>();
    // AscendC::printf("[kfm123] aivId %d\n", aivId_);
    for (uint32_t i = 0; i < compResultU64Num; i++) { 
        uint64_t flagCompMask = flagCompResultLtU64_(i);
        int64_t firstValidIdx = ScalarGetSFFValue<0>(flagCompMask); // 找到0则表示数据没到
        // AscendC::printf("[kfm124] aivId %d\n", aivId_);
        if (firstValidIdx == -1) { // 本次数据全到
            arriveFlagNum += 64U; // 64：ScalarGetSFFValue操作单位为64bit位
            // AscendC::printf("[kfm125] aivId %d\n", aivId_);
        } else {
            arriveFlagNum += uint32_t(firstValidIdx);
            // AscendC::printf("[kfm126] aivId %d\n", aivId_);
            break;
        }
    }
    // AscendC::printf("[kfm127] aivId %d\n", aivId_);
    // AscendC::printf("[kfm] aivId %d arriveFlagNum %d\n", aivId_, arriveFlagNum);
    if (arriveFlagNum > flagNum) {
        arriveFlagNum = flagNum;
    }
    // AscendC::printf("[kfm] aivId %d arriveFlagNum %d\n", aivId_, arriveFlagNum);
    return uint32_t(arriveFlagNum / blockCntPerToken_); // 返回token总数
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CopyInAndOut(
    LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount, uint32_t dstExpertId)
{
    uint32_t hOutElemCount = hOutSize_ / sizeof(XOutType); // expandXOutGlobal申请每个token的GM Buffer空间大小
    GlobalTensor<XOutType> dataFlagGlobal, expandXOutGlobal;
    dataFlagGlobal.SetGlobalBuffer((__gm__ XOutType *)(wAddr));
    expandXOutGlobal.SetGlobalBuffer((__gm__ XOutType *)(expandXOutGM_) + (dstPosition) * hOutElemCount);
    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerToken_ * arriveCount), 
        static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE), static_cast<uint16_t>(UB_ALIGN), 0};
    DataCopyExtParams scalesCopyParams{uint16_t(arriveCount), static_cast<uint32_t>(scaleOutBytes_), 
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - scaleOutBytes_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams tokenCopyParams{uint16_t(arriveCount), hOutSize_, 
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams expandIdxCopyParams{uint16_t(arriveCount), EXPAND_IDX_INFO * sizeof(int32_t),
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};

    DataCopyPad(xTmpTensor_, dataFlagGlobal[expertFinishNumTensor_(index) * hCommuSize_ / sizeof(XOutType)],
                srcTokenCopyParams, srcTokenPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    quantInst_.CopyScalesToOut(dstPosition, scaleOutBytes_, xTmpTensor_, scalesCopyParams);
    DataCopyPad(expandXOutGlobal, xTmpTensor_, tokenCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // hzy: URMA 二次转发后，原始 fill 的 topKIndex 固定为 0，需要根据携带的 top-k expert id 修正。
    uint32_t tokenCommCnt = blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE / sizeof(int32_t);
    for (uint32_t i = 0; i < arriveCount; i++) {
        uint32_t topKInfoOffset = i * tokenCommCnt + tokenQuantAlign_ + UB_ALIGN_DATA_COUNT;
        // AscendC::printf("[kfm] CopyInAndOut: aivId_ %d arriveCount %d i %d x(0) %d triple %d %d %d\n", 
            // aivId_, arriveCount, i, xOutInt32Tensor(i * tokenCommCnt), xOutInt32Tensor(i * tokenCommCnt + tokenQuantAlign_),
            // xOutInt32Tensor(i * tokenCommCnt + tokenQuantAlign_ + 1), xOutInt32Tensor(i * tokenCommCnt + tokenQuantAlign_ + 2));
        for (uint32_t j = 0; j < axisK_; j++) {
            int32_t curExpertId = xOutInt32Tensor(topKInfoOffset + j);
            if (dstExpertId == static_cast<uint32_t>(curExpertId)) {
                xOutInt32Tensor(tokenQuantAlign_ + 2 + i * tokenCommCnt) = j;
                break;
            }
        }
        // AscendC::printf("[kfm] CopyInAndOut final: aivId_ %d arriveCount %d i %d x(0) %d triple %d %d %d\n",
        //     aivId_, arriveCount, i, xOutInt32Tensor(i * tokenCommCnt), xOutInt32Tensor(i * tokenCommCnt + tokenQuantAlign_),
        //     xOutInt32Tensor(i * tokenCommCnt + tokenQuantAlign_ + 1), xOutInt32Tensor(i * tokenCommCnt + tokenQuantAlign_ + 2));
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(expandIdxGMTensor_[dstPosition * EXPAND_IDX_INFO], xOutInt32Tensor[tokenQuantAlign_],
                expandIdxCopyParams);
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum)
{
    // AscendC::printf("[kfm10] aivId %d\n", aivId_);
    uint32_t index = 0;
    uint32_t finishNum = 0;
    uint32_t gatherOutSizePerToken =  Ceil(blockCntPerToken_ * sizeof(uint32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    uint32_t maxCopyTokenCnt = tBufRealSize_ / (hCommuSize_ + gatherOutSizePerToken);
    // AscendC::printf("[kfm] aivId_ %d maxCopyTokenCnt %d\n", aivId_, maxCopyTokenCnt);
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    uint32_t srcExpRankId, dstPosition, arriveCount, copyCnt, srcDataBlockIdx;
    uint32_t flagMaxRecvSize = blockCntPerToken_ * maxCopyTokenCnt * UB_ALIGN;
    uint32_t flagMaxRecvNum = flagMaxRecvSize / sizeof(float);
    // AscendC::printf("[kfm] aivId_ %d flagMaxRecvNum %d\n", aivId_, flagMaxRecvNum);
    uint32_t gatherOutSize = Ceil(blockCntPerToken_ * maxCopyTokenCnt * sizeof(uint32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    uint32_t xTmpSize = blockCntPerToken_ * maxCopyTokenCnt * SPLIT_BLOCK_DATA_SIZE;
    GlobalTensor<float> cleanGlobal;
    flagGatherOutTensor_ = tBuf.GetWithOffset<float>(gatherOutSize / sizeof(float), 0); 
    flagRecvTensor_ = tBuf.GetWithOffset<float>(flagMaxRecvNum, gatherOutSize);
    xTmpTensor_ = tBuf.GetWithOffset<XOutType>(xTmpSize / sizeof(XOutType), gatherOutSize + flagMaxRecvSize);
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor_.template ReinterpretCast<int32_t>();
    DataCopyParams cleanUpParams = {uint16_t(blockCntPerToken_), 1U, 0U, SPLIT_BLOCK_DATA_SIZE / UB_ALIGN};
    // AscendC::printf("[kfm11] aivId %d\n", aivId_);


    while (true) {
        if (expertLeftNumTensor_(index) == 0) { // 当前核负责的不需要收集
            index = (index + 1) % validNum; // 轮询查询每个有效的index
            continue;
        }
        srcExpRankId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ? maxCopyTokenCnt : expertLeftNumTensor_(index); // 按照ub大小一次搬入多个token
        srcDataBlockIdx = srcExpRankId % epWorldSize_ * localExpertNum + srcExpRankId / epWorldSize_; // 转换成数据区的排布偏移
        // AscendC::printf("[kfm12] aivId %d\n", aivId_);

        GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_;

        // AscendC::printf("[kfm] CheckAddr aivId %d srcExpRankId %d srcRank %d localExpId %d srcDataBlockIdx %d wAddr %p expertPerSizeOnWin_ %d\n", 
        //     aivId_, srcExpRankId, srcExpRankId % epWorldSize_, srcExpRankId / epWorldSize_, srcDataBlockIdx, wAddr, expertPerSizeOnWin_
        //     );

        // AscendC::printf("[recvAddr] rank %d aiv %d srcExpRankId %u block %u windowGM %p wAddr %p left %u finish %u copyCnt %u\n",
        //     epRankId_, aivId_, srcExpRankId, srcDataBlockIdx, windowGM_, wAddr,
        //     expertLeftNumTensor_(index), expertFinishNumTensor_(index), copyCnt);

        // AscendC::printf("[kfm] aivId %d index %d expertFinishNumTensor_(index) %d copyCnt %d\n", aivId_, index, expertFinishNumTensor_(index), copyCnt);
        arriveCount = CheckDataArriveWithFlag(srcDataBlockIdx, expertFinishNumTensor_(index), copyCnt);
        // AscendC::printf("[kfm13] aivId %d\n", aivId_);
        // AscendC::printf("[kfm] aivId %d index %d arriveCount %d\n", aivId_, index, arriveCount);
        
        if (arriveCount == copyCnt) {
            dstPosition = srcExpRankId != 0 ? sendCntTensor_(srcExpRankId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);

            uint32_t dstExpertId = srcExpRankId / epWorldSize_ + epRankId_ * localExpertNum;
            // AscendC::printf("[kfm14] aivId %d\n", aivId_);
            CopyInAndOut(xOutInt32Tensor, wAddr, index, dstPosition, arriveCount, dstExpertId);
            // AscendC::printf("[kfm15] aivId %d\n", aivId_);
            // finish更新并clean
            expertFinishNumTensor_(index) += arriveCount;
            expertLeftNumTensor_(index) -= arriveCount;
            PipeBarrier<PIPE_ALL>();
            if (expertLeftNumTensor_(index) == 0) {
                cleanGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));
                for (uint32_t i = 0; i < expertFinishNumTensor_(index); i++){
                    uint32_t flagIndex = i * SPLIT_BLOCK_COUNT * blockCntPerToken_ + SPLIT_BLOCK_DATA_COUNT;
                    DataCopy(cleanGlobal[flagIndex], cleanUpTensor_, cleanUpParams);
                }
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                finishNum++;
            }
            // AscendC::printf("[kfm16] aivId %d\n", aivId_);
        } else {
            index = (index + 1) % validNum;
        }
        // AscendC::printf("[kfm17] aivId %d\n", aivId_);
        if (validNum == finishNum) {
            break;
        }
        // break;
        // AscendC::printf("[kfm18] aivId %d\n", aivId_);
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::RunPosRecord(const uint32_t runPos)
{
    TBuf<> runPosBuf;
    tpipe_->InitBuffer(runPosBuf, UB_ALIGN);
    dataStateLocalTensor_ = runPosBuf.Get<uint32_t>();
    dataStateLocalTensor_.SetValue(0, runPos);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams_);    // 维测打点
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::LocalWindowCopy()
{
    // 分核负责源专家数量
    tpipe_->Reset();
    TBuf<> cumSumBuf, statusWaitBuf, statusCleanBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusWaitBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(statusCleanBuf, aivUsedCumSum_ * UB_ALIGN);
    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    statusCleanFp32Tensor_ = statusCleanBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();
    SplitToCore(rscvStatusNum_, aivNum_, startId_, endId_, sendNum_, true);
    // 软同步
    WaitCumSumFlag();
    if (sendNum_ == 0) {
        return;
    }
    // 连续化
    TBuf<> expertMapBuf, expertFinishBuf, expertLeftBuf, flagMaskBuf, cleanUpBuf, tBuf;
    uint32_t validNum = 0;
    uint32_t expInfoSize = Ceil(sendNum_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);
    tpipe_->InitBuffer(flagMaskBuf, BUFFER_NUM * UB_ALIGN);  // max CompareScalar
    tpipe_->InitBuffer(cleanUpBuf, blockCntPerToken_ * UB_ALIGN);
    tBufRealSize_ = MAX_UB_SIZE - (UB_ALIGN + rscvNumAlign + 2 * aivUsedCumSum_ * UB_ALIGN) -
        (expInfoSize * 3) - BUFFER_NUM * UB_ALIGN - blockCntPerToken_ * UB_ALIGN; // 3为expInfoSize大小buffer申请个数
    // AscendC::printf("[kfm] aivId_ %d tBufRealSize_ %d\n", aivId_, tBufRealSize_);
    tpipe_->InitBuffer(tBuf, tBufRealSize_); // 其余buffer空间统一申请
    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();
    SetValidExpertInfo(expInfoSize, validNum);
    // AscendC::printf("[kfm] LocalWindowCopy validNum: aivId_ %d startId_ %d endId_ %d validNum %d\n", 
        // aivId_, startId_, endId_, validNum);
    if (validNum == 0) { // 本核负责的Expert对应rank收到数据
        return;
    }
    flagCompResultU8_ = flagMaskBuf.Get<uint8_t>();
    flagCompResultLtU64_ = flagMaskBuf.Get<uint64_t>();
    flagRecvGatherMask_ = statusCleanBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    cleanUpTensor_ = cleanUpBuf.Get<float>();
    // xTmpTensor_ = tBuf.Get<XOutType>();
    LocalTensor<uint32_t> flagCompResultLtU32 = flagMaskBuf.Get<uint32_t>();
    Duplicate<uint32_t>(flagCompResultLtU32, 0, BUFFER_NUM * UB_ALIGN / sizeof(uint32_t));
    Duplicate<uint32_t>(flagRecvGatherMask_, 0, UB_ALIGN / sizeof(uint32_t));
    Duplicate<float>(cleanUpTensor_, float(0), blockCntPerToken_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    flagRecvGatherMask_.SetValue(0, 1);
    SyncFunc<AscendC::HardEvent::S_V>();
    WaitAndFormatOutput(tBuf, validNum);
    // RunPosRecord(RUNPOS_ARRIVECNT); // 维测打点
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::TokenActiveMaskCal()
{
    // 搬运x_active_mask，当前仅用于计算有效token总数
    LocalTensor<half> maskTmpTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    uint32_t axisBsAlignSize = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    maskInputTensor = dstExpBuf_.Get<bool>();
    maskTmpTensor = subExpBuf_.Get<half>();
    sumOutTensor = gatherMaskTBuf_.Get<half>();
    DataCopyExtParams maskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(maskTmpTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize, axisBS_};
    Sum(sumOutTensor, maskTmpTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalValidBSCnt(LocalTensor<bool> maskStrideTensor)
{
    uint64_t rsvdCnt = 0;
    uint32_t mask = axisBS_;
    uint32_t activeMaskAlignSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    LocalTensor<half> tempTensor = validExpertIndexBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = expertIdsBuf_.Get<half>();
    LocalTensor<half> tokenTargetTensor = validBsIndexTBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = subExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = gatherMaskTBuf_.Get<uint32_t>();
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    SumParams axisBsSumParams{1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize);
    PipeBarrier<PIPE_V>();
    Sum(tokenTargetTensor, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(validBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalValidExpIdx(LocalTensor<bool> maskInputTensor)
{
    uint32_t mask = expertIdsCnt_;
    uint32_t curMaskCnt = axisBS_ * axisK_;
    uint32_t calCnt = Ceil(curMaskCnt * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    LocalTensor<int32_t> validExpertIndexTensor = validExpertIndexBuf_.Get<int32_t>();
    LocalTensor<half> tempTensor = subExpBuf_.Get<half>();
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> expertsIndexTensor = expertIdsBuf_.Get<int32_t>();

    Duplicate<half>(tempTensor, (half)0, calCnt);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, curMaskCnt);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, SIZE_ALIGN_256) * SIZE_ALIGN_256 / BITS_PER_BYTE / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, curMaskCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0}, sendToMoeExpTokenCnt_);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::ExpertActiveMaskInit()
{
    uint32_t axisBSAlign = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    tpipe_->InitBuffer(validBsIndexTBuf_, axisBSAlign);
    uint32_t validBufferSize = expertIdsSize_ > xActivateMaskSize ? expertIdsSize_ : xActivateMaskSize;
    tpipe_->InitBuffer(validExpertIndexBuf_, validBufferSize);
    validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::ExpertActiveMaskCal()
{
    // 计算当前有效bs数量, stride搬入xActiveMask进行sum计算, 用于moe专家发送
    LocalTensor<bool> maskStrideTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskStrideCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskStrideParams{
        static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(maskStrideTensor, xActiveMaskGMTensor_, maskStrideParams, maskStrideCopyPadParams);
    CalValidBSCnt(maskStrideTensor);
    // 计算validExpIndexTensor, 连续搬入xActiveMask进行GatherMask计算, 用于moe专家的发送
    LocalTensor<bool> maskInputTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    CalValidExpIdx(maskInputTensor);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::MaskZeroComputeExpert(uint32_t maskCnt)
{
    LocalTensor<int32_t> expertsIndexTensor = dstExpBuf_.Get<int32_t>();
    int32_t maskTensorInt16Cnt = Ceil(expertIdsCnt_, UB_ALIGN / 2);
    LocalTensor<uint8_t> maskTensorInt8 = validExpertIndexBuf_.Get<uint8_t>(); // bs*k*1
    LocalTensor<uint32_t> maskTensorInt32 = validExpertIndexBuf_.Get<uint32_t>(); // bs*k*1
    LocalTensor<half> expertIdsTensorCast = subExpBuf_.Get<half>(); // bs*k*4
    LocalTensor<int32_t> validExpertIndexTensor = validExpertIndexBuf_.Get<int32_t>();
    int32_t moeExpertNumInt32 = static_cast<int32_t>(moeExpertNum_);

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(expertIdsTensorCast, validExpertIdsTensor_, RoundMode::CAST_NONE, expertIdsCnt_);
    Duplicate<uint32_t>(maskTensorInt32, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    uint32_t calcCnt = Ceil(expertIdsCnt_ * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    // 逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。筛掉零计算量专家
    CompareScalar(maskTensorInt8, expertIdsTensorCast, static_cast<half>(moeExpertNumInt32), AscendC::CMPMODE::LT, calcCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint16_t> maskTensorInt16 = validExpertIndexBuf_.Get<uint16_t>(); // 空间bs*k*1
    LocalTensor<uint16_t> gatherMaskTensorint16 = gatherMaskTBuf_.Get<uint16_t>(); // 空间bs*k*4
    /* 特殊专家的maskTensorInt16和之前的gatherMaskTensor_结果按位相与，AND 支持uint16， gatherMaskTensor_和gatherMaskTensorint16是同一个地址 */
    And(gatherMaskTensorint16, gatherMaskTensorint16, maskTensorInt16, maskTensorInt16Cnt);
    PipeBarrier<PIPE_V>();
    // 再筛一次
    CreateVecIndex(expertsIndexTensor, 0, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor_, true, maskCnt, {1, 1, 0, 0}, sendToMoeExpTokenCnt_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::GenerateGatherMaskTensor(uint32_t maskCnt)
{
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0xFFFFFFFF, Ceil(maskCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::ZeroComputeExpertMaskCal()
{
    uint32_t maskCnt = expertIdsCnt_;
    if (isTokenMaskFlag_) { // 一维
        maskCnt = activeMaskBsCnt_ * axisK_;
    }

    if (!isExpertMaskFlag_) { // 非二维要生成gatherMaskTensor_
        GenerateGatherMaskTensor(maskCnt);
    }

    // 零计算量专家剪枝
    MaskZeroComputeExpert(maskCnt);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::ExpIdsCopyAndMaskCal()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    validExpertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        ExpertActiveMaskInit();
    }

    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }

    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }
    if (activeMaskBsCnt_ == 0) { // DataCopyPad不能拷贝0个
        return;
    }
    if (zeroComputeExpertNum_ != 0) {
        ZeroComputeExpertMaskCal();
    }

    Duplicate<int32_t>(validExpertIdsTensor_, -1, int32_t(expertIdsBufSize_ / sizeof(int32_t)));
    // 拷贝bs*k个专家id 到local  补齐到32 字节对齐  bs*k = 3*3 = 9 expertIdsAlignCnt= 16 补7个 填充-1
    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        LocalTensor<int32_t> tmpExpertIdsTensor = subExpBuf_.Get<int32_t>();
        LocalTensor<float> tmpExpertIdsTensorFloat = subExpBuf_.Get<float>();
        LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTensor_.ReinterpretCast<uint8_t>();
        DataCopyExtParams expertIdsMaskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> expertIdsMaskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tmpExpertIdsTensor, expertIdsGMTensor_, expertIdsMaskParams, expertIdsMaskCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_V>();
        LocalTensor<float> validExpertIdsFloat = validExpertIdsTensor_.ReinterpretCast<float>();
        Select(validExpertIdsFloat, gatherMaskTensorInt8, tmpExpertIdsTensorFloat, static_cast<float>(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE, expertIdsCnt_);
        SyncFunc<AscendC::HardEvent::V_S>();
    } else {
        uint32_t expertIdsMask = activeMaskBsCnt_ * axisK_;
        uint32_t expertIdsAlignCnt = Ceil(expertIdsMask, BITS_PER_BYTE) * BITS_PER_BYTE;
        uint32_t rightPadding = expertIdsAlignCnt - expertIdsMask;
        DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{true, 0U, uint8_t(rightPadding), -1}; // rightPadding字节数不能超过 32，不能超过8个u32
        DataCopyExtParams expertIdsCntParams{1U, static_cast<uint32_t>(expertIdsMask * sizeof(uint32_t)), 0U, 0U, 0U}; //第二个参数blockLen 范围[1, 2097151] 不能为0
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
}


struct HcclAiRMAWQ {
    uint8_t rmtEid[16];
    uint32_t jettyId;
    uint64_t sqVA;     // SQE在HBM上起始地址
    uint32_t wqeSize;  // 一个WQEBB占用内存大小（64B）
    uint32_t sqDepth;  // 可用的WQEBB个数
    uint64_t headAddr; // AIV无依赖
    uint64_t tailAddr; // AIV无依赖
    uint64_t dbAddr;   // JFSDoorBell地址
    uint32_t tp_id;
    uint32_t rmtObjId; // rmtTokenID
    uint32_t rmtTokenValue;
    uint32_t localTokenId;
};

struct HcclAiRMACQ {
    uint32_t jfcId;
    uint64_t cqVA;    // CQE在HBM上起始地址
    uint32_t cqeSize; // 一个CQE占用内存大小（64B）
    uint32_t cqDepth; // 可用的CQE个数
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr; // JFCDoorBell地址
};

enum class URMAOPCODE : uint32_t {
    OP_SEND = 0,
    OP_SEND_WITH_IMM,
    OP_SEND_WITH_INV,
    OP_WRITE,
    OP_WRITE_WITH_IMM,
    OP_WRITE_WITH_NOTIFY,
    OP_READ
};

typedef HcclAiRMAWQ AiURMAWQ;
typedef HcclAiRMACQ AiURMACQ; 

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::get_sqs_and_cqs_xb(GM_ADDR sqs, GM_ADDR cqs)
{
	uint32_t rankNum = aclshmem_n_pes();
    // AscendC::printf("[kfm] aivId_ %d rankNum %d\n", aivId_, rankNum);
	__gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();

	// AscendC::printf("udma info %p, sqs %p, cqs %p, mem %p\n", udmaInfo, udmaInfo->sqPtr, udmaInfo->scqPtr, udmaInfo->memPtr);

	// Get data in the "rank -> port -> jetty" order
	for (uint32_t rankIdx = 0; rankIdx < rankNum; rankIdx++) {
		if (rankIdx == aclshmem_my_pe()) {
			continue;
		}

		for (uint32_t portIdx = 0; portIdx < PORT_NUM; portIdx++) {
			for (uint32_t jettyIdx = 0; jettyIdx < JETTY_NUM; jettyIdx++) {
				auto offset = rankIdx * PORT_NUM * JETTY_NUM + portIdx * JETTY_NUM + jettyIdx;

				__gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry =
					(__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr + offset * sizeof(ACLSHMEMUDMAWQCtx));
				__gm__ ACLSHMEMUDMACqCtx* cqCtxEntry =
					(__gm__ ACLSHMEMUDMACqCtx*)(udmaInfo->scqPtr + offset * sizeof(ACLSHMEMUDMACqCtx));
				__gm__ ACLSHMEMUBmemInfo* remoteMemInfo =
					(__gm__ ACLSHMEMUBmemInfo*)(udmaInfo->memPtr + offset * sizeof(ACLSHMEMUBmemInfo));

				__gm__ HcclAiRMAWQ* hcclAiRMAWQ = (__gm__ HcclAiRMAWQ*)(sqs + offset * sizeof(HcclAiRMAWQ));
                // AscendC::printf("[shmem] portIdx %d jettyIdx %d baseBkShift %d bufAddr %p headAddr %p tailAddr %p dbAddr %p\n", 
                //     portIdx, jettyIdx, qpCtxEntry->baseBkShift, qpCtxEntry->bufAddr, qpCtxEntry->headAddr, qpCtxEntry->tailAddr, qpCtxEntry->dbAddr);
                hcclAiRMAWQ->jettyId = qpCtxEntry->wqn;
				hcclAiRMAWQ->sqVA = qpCtxEntry->bufAddr;
				hcclAiRMAWQ->wqeSize = (uint32_t)(1 << qpCtxEntry->baseBkShift);
				hcclAiRMAWQ->sqDepth = qpCtxEntry->depth;
				hcclAiRMAWQ->headAddr = qpCtxEntry->headAddr;
				hcclAiRMAWQ->tailAddr = qpCtxEntry->tailAddr;
				hcclAiRMAWQ->dbAddr = qpCtxEntry->dbAddr;
				hcclAiRMAWQ->tp_id = remoteMemInfo->tpn;

				// AscendC::printf("[kfm] hcclAiRMAWQ->wqeSize %d sqVA %p headAddr %p tailAddr %p dbAddr %p\n", 
                //     hcclAiRMAWQ->wqeSize, hcclAiRMAWQ->sqVA, hcclAiRMAWQ->headAddr, hcclAiRMAWQ->tailAddr, hcclAiRMAWQ->dbAddr);

				// auto eidAddrList = (__gm__ uint64_t*)(remoteMemInfo->eidAddr);
				// auto rmtEid = (__gm__ uint64_t*)(hcclAiRMAWQ->rmtEid);
				// AscendC::printf("rank %d meminfo remote eid addr %p, eidl %p, eidh %p\n", rankIdx, eidAddrList, eidAddrList[0], eidAddrList[1]);		// OK
				// AscendC::printf("rank %d wq %p, eid addr %p, eidl %p, eidh %p\n", rankIdx, hcclAiRMAWQ, hcclAiRMAWQ->rmtEid, rmtEid[0], rmtEid[1]);	// Seg fault
				// rmtEid[0] = eidAddrList[0];																											// Seg fault
				// rmtEid[1] = eidAddrList[1];																											// Seg fault

				auto eids = (__gm__ uint8_t*)(remoteMemInfo->eidAddr);
				for (int i = 0; i < 16; i++) {
					hcclAiRMAWQ->rmtEid[i] = eids[i];
				}

				hcclAiRMAWQ->rmtObjId = remoteMemInfo->tid;
				hcclAiRMAWQ->rmtTokenValue = remoteMemInfo->rmtTokenValue;
				hcclAiRMAWQ->localTokenId = 0; // NOTE: not use

				// AscendC::printf("[kfm] cqCtxEntry->baseBkShift %d cqn %d bufAddr %p depth %d headAddr %p tailAddr %p dbAddr %p\n", 
                //     cqCtxEntry->baseBkShift, cqCtxEntry->cqn, cqCtxEntry->bufAddr, cqCtxEntry->depth, cqCtxEntry->headAddr,
                //     cqCtxEntry->tailAddr, cqCtxEntry->dbAddr);

				__gm__ HcclAiRMACQ* hcclAiRMACQ = (__gm__ HcclAiRMACQ*)(cqs + offset * sizeof(HcclAiRMACQ));
				hcclAiRMACQ->jfcId = cqCtxEntry->cqn;
				hcclAiRMACQ->cqVA = cqCtxEntry->bufAddr;
				hcclAiRMACQ->cqeSize = (uint32_t)(1 << cqCtxEntry->baseBkShift);
				hcclAiRMACQ->cqDepth = cqCtxEntry->depth;
				hcclAiRMACQ->headAddr = cqCtxEntry->headAddr;
				hcclAiRMACQ->tailAddr = cqCtxEntry->tailAddr;
				hcclAiRMACQ->dbAddr = cqCtxEntry->dbAddr;
				// AscendC::printf("[kfm] hcclAiRMACQ->cqeSize %d jfcId %d cqVA %p cqDepth %d headAddr %p tailAddr %p dbAddr %p\n", 
                //     hcclAiRMACQ->cqeSize, hcclAiRMACQ->jfcId, hcclAiRMACQ->cqVA, hcclAiRMACQ->cqDepth, hcclAiRMACQ->headAddr,
                //     hcclAiRMACQ->tailAddr, hcclAiRMACQ->dbAddr);
			}
		}
	}
}

ACLSHMEM_DEVICE void get_sqs_and_cqs_xb_ub(uint64_t sqs, uint64_t cqs, TPipe *tpipe)
{
    uint32_t rankNum = aclshmem_n_pes();
    uint32_t totalEntries = rankNum * PORT_NUM * JETTY_NUM;

    // 需要3个临时缓冲区用于GM->UB中转；每个entry按32B独立对齐，避免DataCopyPad落到非对齐地址
    TBuf<> sqBuf, cqBuf, WQTempBuf, CQTempBuf, memTempBuf;
    uint32_t alignedWqSize = Ceil(sizeof(ACLSHMEMUDMAWQCtx), UB_ALIGN) * UB_ALIGN;
    uint32_t alignedCqSize = Ceil(sizeof(ACLSHMEMUDMACqCtx), UB_ALIGN) * UB_ALIGN;
    uint32_t alignedMemSize = Ceil(sizeof(ACLSHMEMUBmemInfo), UB_ALIGN) * UB_ALIGN;
    uint32_t wqTempSize = totalEntries * alignedWqSize;
    uint32_t cqTempSize = totalEntries * alignedCqSize;
    uint32_t memTempSize = totalEntries * alignedMemSize;

    tpipe->InitBuffer(WQTempBuf, wqTempSize);  // 临时中转
    tpipe->InitBuffer(CQTempBuf, cqTempSize);  // 临时中转
    tpipe->InitBuffer(memTempBuf, memTempSize); // 临时中转

    // sqs和cqs目标缓冲区（通过GetWithOffset访问）
    //tpipe_->InitBuffer(sqBuf, totalEntries * sizeof(HcclAiRMAWQ));
    //tpipe_->InitBuffer(cqBuf, totalEntries * sizeof(HcclAiRMACQ));
    
    LocalTensor<uint8_t> WQ_LT = WQTempBuf.Get<uint8_t>();
    LocalTensor<uint8_t> CQ_LT = CQTempBuf.Get<uint8_t>();
    LocalTensor<uint8_t> memInfo_LT = memTempBuf.Get<uint8_t>();
    
    //LocalTensor<uint8_t> sqLT = sqBuf.Get<uint8_t>();
    //LocalTensor<uint8_t> cqLT = cqBuf.Get<uint8_t>();

    GlobalTensor<uint8_t> WQ_GM, CQ_GM, memInfo_GM;
    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();

    // GM -> UB临时缓冲区
    for (uint32_t rankIdx = 0; rankIdx < rankNum; rankIdx++) {
        if (rankIdx == aclshmem_my_pe()) {
            continue;
        }

        for (uint32_t portIdx = 0; portIdx < PORT_NUM; portIdx++) {
            for (uint32_t jettyIdx = 0; jettyIdx < JETTY_NUM; jettyIdx++) {
                auto offset = rankIdx * PORT_NUM * JETTY_NUM + portIdx * JETTY_NUM + jettyIdx;

                __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry =
                    (__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr + offset * sizeof(ACLSHMEMUDMAWQCtx));
                __gm__ ACLSHMEMUDMACqCtx* cqCtxEntry =
                    (__gm__ ACLSHMEMUDMACqCtx*)(udmaInfo->scqPtr + offset * sizeof(ACLSHMEMUDMACqCtx));
                __gm__ ACLSHMEMUBmemInfo* remoteMemInfo =
                    (__gm__ ACLSHMEMUBmemInfo*)(udmaInfo->memPtr + offset * sizeof(ACLSHMEMUBmemInfo));

                WQ_GM.SetGlobalBuffer((__gm__ uint8_t*)qpCtxEntry);
                CQ_GM.SetGlobalBuffer((__gm__ uint8_t*)cqCtxEntry);
                memInfo_GM.SetGlobalBuffer((__gm__ uint8_t*)remoteMemInfo);

                DataCopyParams paramsWQCtx{1U,
                    static_cast<uint16_t>(sizeof(ACLSHMEMUDMAWQCtx)), 0U, 0U};
                DataCopyPad(WQ_LT[offset * alignedWqSize], WQ_GM, paramsWQCtx, DataCopyPadParams{false, 0U, 0U, 0U});

                DataCopyParams paramsCQCtx{1U,
                    static_cast<uint16_t>(sizeof(ACLSHMEMUDMACqCtx)), 0U, 0U};
                DataCopyPad(CQ_LT[offset * alignedCqSize], CQ_GM, paramsCQCtx, DataCopyPadParams{false, 0U, 0U, 0U});

                DataCopyParams paramsMemInfo{1U,
                    static_cast<uint16_t>(sizeof(ACLSHMEMUBmemInfo)), 0U, 0U};
                DataCopyPad(memInfo_LT[offset * alignedMemSize], memInfo_GM, paramsMemInfo, DataCopyPadParams{false, 0U, 0U, 0U});
                SyncFunc<AscendC::HardEvent::MTE2_S>();
            }
        }
    }

    // SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_V>();
   // 赋值sqs和cqs
    for (uint32_t rankIdx = 0; rankIdx < rankNum; rankIdx++) {
        if (rankIdx == aclshmem_my_pe()) continue;
        for (uint32_t portIdx = 0; portIdx < PORT_NUM; portIdx++) {
            for (uint32_t jettyIdx = 0; jettyIdx < JETTY_NUM; jettyIdx++) {
                uint64_t offset = uint64_t(rankIdx * PORT_NUM * JETTY_NUM + portIdx * JETTY_NUM + jettyIdx);

                uint64_t wqOffset = offset * alignedWqSize;
                uint64_t cqOffset = offset * alignedCqSize;
                uint64_t memOffset = offset * alignedMemSize;

                // 获取UB上的结构体指针，直接用->赋值
                __ubuf__ ACLSHMEMUDMAWQCtx* wqCtx = (__ubuf__ ACLSHMEMUDMAWQCtx*)(reinterpret_cast<uint64_t>(WQ_LT.GetPhyAddr()) + wqOffset);
                __ubuf__ ACLSHMEMUDMACqCtx* cqCtx = (__ubuf__ ACLSHMEMUDMACqCtx*)(reinterpret_cast<uint64_t>(CQ_LT.GetPhyAddr()) + cqOffset);
                __ubuf__ ACLSHMEMUBmemInfo* memCtx = (__ubuf__ ACLSHMEMUBmemInfo*)(reinterpret_cast<uint64_t>(memInfo_LT.GetPhyAddr()) + memOffset);

                __ubuf__ HcclAiRMAWQ* hcclAiRMAWQ = (__ubuf__ HcclAiRMAWQ*)(sqs + offset * sizeof(HcclAiRMAWQ));
                __ubuf__ HcclAiRMACQ* hcclAiRMACQ = (__ubuf__ HcclAiRMACQ*)(cqs + offset * sizeof(HcclAiRMACQ));

                // 直接用->赋值
                hcclAiRMAWQ->jettyId = wqCtx->wqn;
                hcclAiRMAWQ->sqVA = wqCtx->bufAddr;
                hcclAiRMAWQ->wqeSize = (uint32_t)(1 << wqCtx->baseBkShift);
                hcclAiRMAWQ->sqDepth = wqCtx->depth;
                hcclAiRMAWQ->headAddr = wqCtx->headAddr;
                hcclAiRMAWQ->tailAddr = wqCtx->tailAddr;
                hcclAiRMAWQ->dbAddr = wqCtx->dbAddr;
                hcclAiRMAWQ->tp_id = memCtx->tpn;
                
                auto eids = (__ubuf__ uint8_t*)(memCtx->eidAddr);
                for (int i = 0; i < 16; i++) {
                    hcclAiRMAWQ->rmtEid[i] = eids[i];
                }
                
                hcclAiRMAWQ->rmtObjId = memCtx->tid;
                hcclAiRMAWQ->rmtTokenValue = memCtx->rmtTokenValue;
                hcclAiRMAWQ->localTokenId = 0;

                hcclAiRMACQ->jfcId = cqCtx->cqn;
                hcclAiRMACQ->cqVA = cqCtx->bufAddr;
                hcclAiRMACQ->cqeSize = (uint32_t)(1 << cqCtx->baseBkShift);
                hcclAiRMACQ->cqDepth = cqCtx->depth;
                hcclAiRMACQ->headAddr = cqCtx->headAddr;
                hcclAiRMACQ->tailAddr = cqCtx->tailAddr;
                hcclAiRMACQ->dbAddr = cqCtx->dbAddr;
            }
        }
    }
}
// get_sqs_and_cqs_xb_ub_v2: GM -> UB1 -> UB2, UB1 has the same data arrangement way with GM
ACLSHMEM_DEVICE void get_sqs_and_cqs_xb_ub_v2(uint64_t sqs, uint64_t cqs, TPipe *tpipe)
{
    uint32_t rankNum = aclshmem_n_pes();
    uint32_t totalEntries = rankNum * PORT_NUM * JETTY_NUM;

    // 需要3个临时缓冲区用于GM->UB中转；每个entry按32B独立对齐，避免DataCopyPad落到非对齐地址
    TBuf<> sqBuf, cqBuf, WQTempBuf, CQTempBuf, memTempBuf;

    uint32_t wqTempSize = Ceil(totalEntries * sizeof(ACLSHMEMUDMAWQCtx), UB_ALIGN) * UB_ALIGN;
    uint32_t cqTempSize = Ceil(totalEntries * sizeof(ACLSHMEMUDMACqCtx), UB_ALIGN) * UB_ALIGN;
    uint32_t memTempSize = Ceil(totalEntries * sizeof(ACLSHMEMUBmemInfo), UB_ALIGN) * UB_ALIGN;

    tpipe->InitBuffer(WQTempBuf, wqTempSize);  // 临时中转
    tpipe->InitBuffer(CQTempBuf, cqTempSize);  // 临时中转
    tpipe->InitBuffer(memTempBuf, memTempSize); // 临时中转

    // sqs和cqs目标缓冲区（通过GetWithOffset访问）
    //tpipe_->InitBuffer(sqBuf, totalEntries * sizeof(HcclAiRMAWQ));
    //tpipe_->InitBuffer(cqBuf, totalEntries * sizeof(HcclAiRMACQ));
    
    LocalTensor<uint8_t> WQ_LT = WQTempBuf.Get<uint8_t>();
    LocalTensor<uint8_t> CQ_LT = CQTempBuf.Get<uint8_t>();
    LocalTensor<uint8_t> memInfo_LT = memTempBuf.Get<uint8_t>();
    
    //LocalTensor<uint8_t> sqLT = sqBuf.Get<uint8_t>();
    //LocalTensor<uint8_t> cqLT = cqBuf.Get<uint8_t>();

    GlobalTensor<uint8_t> WQ_GM, CQ_GM, memInfo_GM;
    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();

    // GM -> UB临时缓冲区
    for (uint32_t rankIdx = 0; rankIdx < rankNum; rankIdx++) {
        if (rankIdx == aclshmem_my_pe()) {
            continue;
        }

        for (uint32_t portIdx = 0; portIdx < PORT_NUM; portIdx++) {
            for (uint32_t jettyIdx = 0; jettyIdx < JETTY_NUM; jettyIdx++) {
                auto offset = rankIdx * PORT_NUM * JETTY_NUM + portIdx * JETTY_NUM + jettyIdx;

                __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry =
                    (__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr + offset * sizeof(ACLSHMEMUDMAWQCtx));
                __gm__ ACLSHMEMUDMACqCtx* cqCtxEntry =
                    (__gm__ ACLSHMEMUDMACqCtx*)(udmaInfo->scqPtr + offset * sizeof(ACLSHMEMUDMACqCtx));
                __gm__ ACLSHMEMUBmemInfo* remoteMemInfo =
                    (__gm__ ACLSHMEMUBmemInfo*)(udmaInfo->memPtr + offset * sizeof(ACLSHMEMUBmemInfo));

                WQ_GM.SetGlobalBuffer((__gm__ uint8_t*)qpCtxEntry);
                CQ_GM.SetGlobalBuffer((__gm__ uint8_t*)cqCtxEntry);
                memInfo_GM.SetGlobalBuffer((__gm__ uint8_t*)remoteMemInfo);

                DataCopyParams paramsWQCtx{1U,
                    static_cast<uint16_t>(sizeof(ACLSHMEMUDMAWQCtx)), 0U, 0U};
                DataCopyPad(WQ_LT[offset * sizeof(ACLSHMEMUDMAWQCtx)], WQ_GM, paramsWQCtx, DataCopyPadParams{false, 0U, 0U, 0U});

                DataCopyParams paramsCQCtx{1U,
                    static_cast<uint16_t>(sizeof(ACLSHMEMUDMACqCtx)), 0U, 0U};
                DataCopyPad(CQ_LT[offset * sizeof(ACLSHMEMUDMAWQCtx)], CQ_GM, paramsCQCtx, DataCopyPadParams{false, 0U, 0U, 0U});

                DataCopyParams paramsMemInfo{1U,
                    static_cast<uint16_t>(sizeof(ACLSHMEMUBmemInfo)), 0U, 0U};
                DataCopyPad(memInfo_LT[offset * sizeof(ACLSHMEMUBmemInfo)], memInfo_GM, paramsMemInfo, DataCopyPadParams{false, 0U, 0U, 0U});
            }
        }
    }

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_V>();
   // 赋值sqs和cqs
    for (uint32_t rankIdx = 0; rankIdx < rankNum; rankIdx++) {
        if (rankIdx == aclshmem_my_pe()) continue;
        for (uint32_t portIdx = 0; portIdx < PORT_NUM; portIdx++) {
            for (uint32_t jettyIdx = 0; jettyIdx < JETTY_NUM; jettyIdx++) {
                uint64_t offset = uint64_t(rankIdx * PORT_NUM * JETTY_NUM + portIdx * JETTY_NUM + jettyIdx);

                uint64_t wqOffset = offset * sizeof(ACLSHMEMUDMAWQCtx);
                uint64_t cqOffset = offset * sizeof(ACLSHMEMUDMAWQCtx);
                uint64_t memOffset = offset * sizeof(ACLSHMEMUBmemInfo);

                // 获取UB上的结构体指针，直接用->赋值
                __ubuf__ ACLSHMEMUDMAWQCtx* wqCtx = (__ubuf__ ACLSHMEMUDMAWQCtx*)(reinterpret_cast<uint64_t>(WQ_LT.GetPhyAddr()) + wqOffset);
                __ubuf__ ACLSHMEMUDMACqCtx* cqCtx = (__ubuf__ ACLSHMEMUDMACqCtx*)(reinterpret_cast<uint64_t>(CQ_LT.GetPhyAddr()) + cqOffset);
                __ubuf__ ACLSHMEMUBmemInfo* memCtx = (__ubuf__ ACLSHMEMUBmemInfo*)(reinterpret_cast<uint64_t>(memInfo_LT.GetPhyAddr()) + memOffset);

                __ubuf__ HcclAiRMAWQ* hcclAiRMAWQ = (__ubuf__ HcclAiRMAWQ*)(sqs + offset * sizeof(HcclAiRMAWQ));
                __ubuf__ HcclAiRMACQ* hcclAiRMACQ = (__ubuf__ HcclAiRMACQ*)(cqs + offset * sizeof(HcclAiRMACQ));

                // 直接用->赋值
                hcclAiRMAWQ->jettyId = wqCtx->wqn;
                hcclAiRMAWQ->sqVA = wqCtx->bufAddr;
                hcclAiRMAWQ->wqeSize = (uint32_t)(1 << wqCtx->baseBkShift);
                hcclAiRMAWQ->sqDepth = wqCtx->depth;
                hcclAiRMAWQ->headAddr = wqCtx->headAddr;
                hcclAiRMAWQ->tailAddr = wqCtx->tailAddr;
                hcclAiRMAWQ->dbAddr = wqCtx->dbAddr;
                hcclAiRMAWQ->tp_id = memCtx->tpn;
                
                auto eids = (__ubuf__ uint8_t*)(memCtx->eidAddr);
                for (int i = 0; i < 16; i++) {
                    hcclAiRMAWQ->rmtEid[i] = eids[i];
                }
                
                hcclAiRMAWQ->rmtObjId = memCtx->tid;
                hcclAiRMAWQ->rmtTokenValue = memCtx->rmtTokenValue;
                hcclAiRMAWQ->localTokenId = 0;

                hcclAiRMACQ->jfcId = cqCtx->cqn;
                hcclAiRMACQ->cqVA = cqCtx->bufAddr;
                hcclAiRMACQ->cqeSize = (uint32_t)(1 << cqCtx->baseBkShift);
                hcclAiRMACQ->cqDepth = cqCtx->depth;
                hcclAiRMACQ->headAddr = cqCtx->headAddr;
                hcclAiRMACQ->tailAddr = cqCtx->tailAddr;
                hcclAiRMACQ->dbAddr = cqCtx->dbAddr;
            }
        }
    }
}

//get_sqs_and_cqs_xb_ub_v3: GM -> UB
ACLSHMEM_DEVICE void get_sqs_and_cqs_xb_ub_v3(uint64_t sqs, uint64_t cqs)
{
    uint32_t rankNum = aclshmem_n_pes();

    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();

    // GM -> UB临时缓冲区
    for (uint32_t rankIdx = 0; rankIdx < rankNum; rankIdx++) {
        if (rankIdx == aclshmem_my_pe()) {
            continue;
        }
        for (uint32_t portIdx = 0; portIdx < PORT_NUM; portIdx++) {
            for (uint32_t jettyIdx = 0; jettyIdx < JETTY_NUM; jettyIdx++) {
                auto offset = rankIdx * PORT_NUM * JETTY_NUM + portIdx * JETTY_NUM + jettyIdx;

                __gm__ ACLSHMEMUDMAWQCtx* wqCtx =
                    (__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr + offset * sizeof(ACLSHMEMUDMAWQCtx));
                __gm__ ACLSHMEMUDMACqCtx* cqCtx =
                    (__gm__ ACLSHMEMUDMACqCtx*)(udmaInfo->scqPtr + offset * sizeof(ACLSHMEMUDMACqCtx));
                __gm__ ACLSHMEMUBmemInfo* memCtx =
                    (__gm__ ACLSHMEMUBmemInfo*)(udmaInfo->memPtr + offset * sizeof(ACLSHMEMUBmemInfo));

                __ubuf__ HcclAiRMAWQ* hcclAiRMAWQ = (__ubuf__ HcclAiRMAWQ*)(sqs + offset * sizeof(HcclAiRMAWQ));
                __ubuf__ HcclAiRMACQ* hcclAiRMACQ = (__ubuf__ HcclAiRMACQ*)(cqs + offset * sizeof(HcclAiRMACQ));

                // 直接用->赋值
                hcclAiRMAWQ->jettyId = wqCtx->wqn;
                hcclAiRMAWQ->sqVA = wqCtx->bufAddr;
                hcclAiRMAWQ->wqeSize = (uint32_t)(1 << wqCtx->baseBkShift);
                hcclAiRMAWQ->sqDepth = wqCtx->depth;
                hcclAiRMAWQ->headAddr = wqCtx->headAddr;
                hcclAiRMAWQ->tailAddr = wqCtx->tailAddr;
                hcclAiRMAWQ->dbAddr = wqCtx->dbAddr;
                hcclAiRMAWQ->tp_id = memCtx->tpn;
                
                auto eids = (__ubuf__ uint8_t*)(memCtx->eidAddr);
                for (int i = 0; i < 16; i++) {
                    hcclAiRMAWQ->rmtEid[i] = eids[i];
                }
                
                hcclAiRMAWQ->rmtObjId = memCtx->tid;
                hcclAiRMAWQ->rmtTokenValue = memCtx->rmtTokenValue;
                hcclAiRMAWQ->localTokenId = 0;

                hcclAiRMACQ->jfcId = cqCtx->cqn;
                hcclAiRMACQ->cqVA = cqCtx->bufAddr;
                hcclAiRMACQ->cqeSize = (uint32_t)(1 << cqCtx->baseBkShift);
                hcclAiRMACQ->cqDepth = cqCtx->depth;
                hcclAiRMACQ->headAddr = cqCtx->headAddr;
                hcclAiRMACQ->tailAddr = cqCtx->tailAddr;
                hcclAiRMACQ->dbAddr = cqCtx->dbAddr;
                PipeBarrier<PIPE_V>();
            }
        }
    }
}

#include "simt_api/vector_functions.h"
#include "simt_api/device_functions.h"

#define DB_ON_UB
#define WQE_ON_UB
// #define JETTY_ON_UB
// #define WQE_OUT_ON_UB

#ifdef DB_ON_UB
#define DB_TMP_POS __ubuf__
#else 
#define DB_TMP_POS __gm__
#endif

#ifdef WQE_ON_UB
#define WQE_IN_POS __ubuf__
#else
#define WQE_IN_POS __gm__
#endif

#ifdef JETTY_ON_UB
#define JETTY_POS __ubuf__
#else
#define JETTY_POS __gm__
#endif

#ifdef WQE_OUT_ON_UB
#define WQE_OUT_POS __ubuf__
#else
#define WQE_OUT_POS __gm__
#endif

// #define MAX_WQE_CNT 128
typedef struct {
    uint64_t localAddr[MAX_WQE_CNT];
    uint64_t remoteAddr[MAX_WQE_CNT];
    uint32_t dstRank[MAX_WQE_CNT];
    uint32_t messageLen[MAX_WQE_CNT];
    URMAOPCODE opcode[MAX_WQE_CNT];
    uint32_t size;
} wqe_desc_array_t;

__simt_callee__ inline void get_wqe_content(WQE_IN_POS wqe_desc_array_t *wqe_desc_arr, uint32_t idx, 
    uint64_t &localAddr, uint64_t &remoteAddr, uint32_t &messageLen, uint32_t &dstRank, URMAOPCODE &op) {
    localAddr = wqe_desc_arr->localAddr[idx];
    remoteAddr = wqe_desc_arr->remoteAddr[idx];
    dstRank = wqe_desc_arr->dstRank[idx];
    messageLen = wqe_desc_arr->messageLen[idx];
    op = wqe_desc_arr->opcode[idx];
}

// #define MAX_WQE_NUM             (256 * 6)
// #define WQE_IN_OFFSET           (0)
// #define WQE_IN_SIZE             (sizeof(wqe_desc_array_t))

// #define MAX_JETTY_NUM           (8 * 2 * 1)
// #define DB_ARRAY_OFFSET         (64 * 1024)
// #define DB_ARRAY_SIZE           (MAX_JETTY_NUM * 8)
// #define HEAD_ARRAY_OFFSET       (DB_ARRAY_OFFSET + DB_ARRAY_SIZE)
// #define HEAD_ARRAY_SIZE         (MAX_JETTY_NUM * 4)
// #define PREV_HEAD_ARRAY_OFFSET  (HEAD_ARRAY_OFFSET + HEAD_ARRAY_SIZE)
// #define PREV_HEAD_ARRAY_SIZE    (MAX_JETTY_NUM * 4)
// #define OUT_JETTY_IDX_OFFSET    (PREV_HEAD_ARRAY_OFFSET + PREV_HEAD_ARRAY_SIZE)
// #define OUT_JETTY_IDX_SIZE      (4)
// #define PORT_IDX_OFFSET         (OUT_JETTY_IDX_OFFSET + OUT_JETTY_IDX_SIZE)
// #define PORT_IDX_SIZE           (4)
// #define JETTY_IDX_OFFSET        (PORT_IDX_OFFSET + PORT_IDX_SIZE)
// #define JETTY_IDX_SIZE          (4)
// #define TOTAL_SIZE              (DB_ARRAY_SIZE + HEAD_ARRAY_SIZE + PREV_HEAD_ARRAY_SIZE + OUT_JETTY_IDX_SIZE + PORT_IDX_SIZE + JETTY_IDX_SIZE)

#define JETTY_INFO_OFFSET       (96 * 1024)
#define JETTY_INFO_SIZE         (MAX_JETTY_NUM * sizeof(HcclAiRMAWQ))

#define WQE_OUT_OFFSET          (120 * 1024)
#define WQE_OUT_SIZE            (MAX_WQE_NUM * 64)

__simt_callee__ inline void simt_write_wqe_v5(JETTY_POS HcclAiRMAWQ *sq, uint32_t myHead, uint64_t localAddr, uint64_t remoteAddr, uint32_t messageLen, URMAOPCODE opcode)
{   
    auto shift = 14;
    auto wqeSize = sq->wqeSize;
    auto qpDepth = sq->sqDepth;

    auto sqBaseAddr = sq->sqVA;
    auto tpId = (sq->tp_id) & ((1U << 24U) - 1U);
    auto rmtObjId = (sq->rmtObjId) & 0xFFFFF;
    auto tokenId = (sq->localTokenId) & 0xFFFFF;
	
    __gm__ ulonglong4* wqeWords = (__gm__ ulonglong4*)(sqBaseAddr + wqeSize * (myHead % qpDepth));

    uint64_t dw[8];
    // Word 0: sqeBbIdx[15:0] | flag[23:16] | rsv0[26:24] | nf[27] | tokenEn[28] | rmtJettyType[30:29] | owner[31]
    // Word 1: targetHint[7:0] | opcode[15:8] | rsv1[21:16] | inlineMsgLen[31:22]
    auto owner = (myHead & qpDepth) == 0 ? 1 : 0;
    dw[0] = (0u << 0)                    // sqeBbIdx = 0
                | (0b00100000u << 16)          // flag
                | (0u << 24)                    // rsv0
                | (0u << 27)                    // nf
                | (1u << 28)     				// tokenEn
                | (1u << 29)          			// rmtJettyType
                | ((uint32_t)owner << 31)       // owner
                | (static_cast<uint32_t>(URMAOPCODE::OP_WRITE) << 40)
                | (0u << 48)                    // rsv1
                | (0u << 54);                   // inlineMsgLen

    // Word 2: tpId[23:0] | sgeNum[31:24]
    // Word 3: rmtJettyOrSegId[19:0] | rsv2[31:20]
    dw[1] = (tpId << 0)
                | (1u << 24)                    // sgeNum = 1
                | (static_cast<uint32_t>(rmtObjId) << 32)
                | (0u << 52);                   // rsv2

    // // Word 4-5: rmtEidL (uint64_t)
    // // Word 6-7: rmtEidH (uint64_t)
	// auto rmtEid = (JETTY_POS uint64_t *)(sq->rmtEid);
    // dw[2] = rmtEid[0];
    // dw[3] = rmtEid[1];
    
    // Word 4-5: rmtEidL (uint64_t)
    // Word 6-7: rmtEidH (uint64_t)
	auto rmtEid = (JETTY_POS uint32_t *)(sq->rmtEid);
    dw[2] = rmtEid[0] | ((rmtEid[1]) << 32);
    dw[3] = rmtEid[2] | ((rmtEid[3]) << 32);

    // Word 8: rmtTokenValue
    // Word 9: udfType[7:0] | reduceDataType[11:8] | reduceOpcode[15:12] | rsv3[31:16]
    dw[4] = 0;

    // Word 10: rmtAddrLOrTokenId
    // Word 11: rmtAddrHOrTokenValue
    dw[5] = remoteAddr;
   
	// SGE
    dw[6] = messageLen;
    dw[7] = localAddr;

    wqeWords[0] = make_ulonglong4(dw[0], dw[1], dw[2], dw[3]);
    wqeWords[1] = make_ulonglong4(dw[4], dw[5], dw[6], dw[7]);
}

__simt_callee__ inline void simt_write_wqe_v4(JETTY_POS HcclAiRMAWQ *sq, uint32_t myHead, uint64_t localAddr, uint64_t remoteAddr, uint32_t messageLen, URMAOPCODE opcode)
{   
    // auto shift = 14;
    auto wqeSize = sq->wqeSize;
    auto qpDepth = sq->sqDepth;

    auto sqBaseAddr = sq->sqVA;
    auto tpId = (sq->tp_id) & ((1U << 24U) - 1U);
    auto rmtObjId = (sq->rmtObjId) & 0xFFFFFU;
    // auto tokenId = (sq->localTokenId) & 0xFFFFF;
	
    __gm__ ulonglong4* wqeWords = (__gm__ ulonglong4*)(sqBaseAddr + wqeSize * (myHead % qpDepth));

    uint64_t dw[8];
    // Word 0: sqeBbIdx[15:0] | flag[23:16] | rsv0[26:24] | nf[27] | tokenEn[28] | rmtJettyType[30:29] | owner[31]
    // Word 1: targetHint[7:0] | opcode[15:8] | rsv1[21:16] | inlineMsgLen[31:22]
    auto owner = (myHead & qpDepth) == 0U ? 1U : 0U;
    dw[0] = (0U << 0U)                    // sqeBbIdx = 0
                | (0b00100000U << 16U)          // flag
                | (0U << 24U)                    // rsv0
                | (0U << 27U)                    // nf
                | (1U << 28U)     				// tokenEn
                | (1U << 29U)          			// rmtJettyType
                | (static_cast<uint32_t>(owner) << 31U)       // owner
                | (static_cast<uint64_t>(opcode) << 40U)
                | (0ULL << 48U)                    // rsv1
                | (0ULL << 54U);                   // inlineMsgLen

    // Word 2: tpId[23:0] | sgeNum[31:24]
    // Word 3: rmtJettyOrSegId[19:0] | rsv2[31:20]
    dw[1] = (tpId << 0U)
                | (1U << 24U)                    // sgeNum = 1
                | (static_cast<uint64_t>(rmtObjId) << 32U)
                | (0ULL << 52U);                   // rsv2

    // // Word 4-5: rmtEidL (uint64_t)
    // // Word 6-7: rmtEidH (uint64_t)
	// auto rmtEid = (JETTY_POS uint64_t *)(sq->rmtEid);
    // dw[2] = rmtEid[0];
    // dw[3] = rmtEid[1];
    
    // Word 4-5: rmtEidL (uint64_t)
    // Word 6-7: rmtEidH (uint64_t)
	auto rmtEid = (JETTY_POS uint32_t *)(sq->rmtEid);
    dw[2] = rmtEid[0] | (static_cast<uint64_t>(rmtEid[1]) << 32U);
    dw[3] = rmtEid[2] | (static_cast<uint64_t>(rmtEid[3]) << 32U);

    // Word 8: rmtTokenValue
    // Word 9: udfType[7:0] | reduceDataType[11:8] | reduceOpcode[15:12] | rsv3[31:16]
    dw[4] = 0U;

    // Word 10: rmtAddrLOrTokenId
    // Word 11: rmtAddrHOrTokenValue
    dw[5] = remoteAddr;
   
	// SGE
    dw[6] = messageLen;
    dw[7] = localAddr;

    wqeWords[0] = make_ulonglong4(dw[0], dw[1], dw[2], dw[3]);
    wqeWords[1] = make_ulonglong4(dw[4], dw[5], dw[6], dw[7]);
}

__simt_callee__ inline void simt_write_wqe_v3(JETTY_POS HcclAiRMAWQ *sq, uint32_t myHead, uint64_t localAddr, uint64_t remoteAddr, uint32_t messageLen, URMAOPCODE opcode)
{
    // constexpr uint32_t shift = 14;
    auto wqeSize = sq->wqeSize;
    auto qpDepth = sq->sqDepth;

    auto sqBaseAddr = sq->sqVA;
    auto tpId = (sq->tp_id) & ((1U << 24U) - 1U);
    auto rmtObjId = (sq->rmtObjId) & 0xFFFFFU;
    // auto tokenId = (sq->localTokenId) & 0xFFFFF;
	
    __gm__ uint64_t* wqeWords = (__gm__ uint64_t*)(sqBaseAddr + wqeSize * (myHead % qpDepth));

    uint64_t dw[8];
    // Word 0: sqeBbIdx[15:0] | flag[23:16] | rsv0[26:24] | nf[27] | tokenEn[28] | rmtJettyType[30:29] | owner[31]
    // Word 1: targetHint[7:0] | opcode[15:8] | rsv1[21:16] | inlineMsgLen[31:22]
    auto owner = (myHead & qpDepth) == 0U ? 1U : 0U;
    dw[0] = (0U << 0U)                    // sqeBbIdx = 0
                | (0b00100000U << 16U)          // flag
                | (0U << 24U)                    // rsv0
                | (0U << 27U)                    // nf
                | (1U << 28U)     				// tokenEn
                | (1U << 29U)          			// rmtJettyType
                | (static_cast<uint32_t>(owner) << 31U)       // owner
                | (static_cast<uint64_t>(opcode) << 40U)
                | (0ULL << 48U)                    // rsv1
                | (0ULL << 54U);                   // inlineMsgLen

    // Word 2: tpId[23:0] | sgeNum[31:24]
    // Word 3: rmtJettyOrSegId[19:0] | rsv2[31:20]
    dw[1] = (tpId << 0U)
                | (1U << 24U)                    // sgeNum = 1
                | (static_cast<uint64_t>(rmtObjId) << 32U)
                | (0ULL << 52U);                   // rsv2

    // // Word 4-5: rmtEidL (uint64_t)
    // // Word 6-7: rmtEidH (uint64_t)
	// auto rmtEid = (JETTY_POS uint64_t *)(sq->rmtEid);
    // dw[2] = rmtEid[0];
    // dw[3] = rmtEid[1];
    
    // Word 4-5: rmtEidL (uint64_t)
    // Word 6-7: rmtEidH (uint64_t)
	auto rmtEid = (JETTY_POS uint32_t *)(sq->rmtEid);
    dw[2] = rmtEid[0] | (static_cast<uint64_t>(rmtEid[1]) << 32U);
    dw[3] = rmtEid[2] | (static_cast<uint64_t>(rmtEid[3]) << 32U);

    // Word 8: rmtTokenValue
    // Word 9: udfType[7:0] | reduceDataType[11:8] | reduceOpcode[15:12] | rsv3[31:16]
    dw[4] = 0U;

    // Word 10: rmtAddrLOrTokenId
    // Word 11: rmtAddrHOrTokenValue
    dw[5] = remoteAddr;
   
	// SGE
    dw[6] = messageLen;
    dw[7] = localAddr;

    #pragma unroll
    for(int i = 0; i < 8; i ++){
        wqeWords[i] = dw[i];
    }
}

__simt_callee__ inline void simt_write_wqe_v2(JETTY_POS HcclAiRMAWQ *sq, uint32_t myHead, uint64_t localAddr, uint64_t remoteAddr, uint32_t messageLen, URMAOPCODE opcode)
{
    constexpr uint32_t shift = 14;
    constexpr uint32_t wqeSize = 64;
    constexpr uint32_t qpDepth = 1 << shift;

    auto sqBaseAddr = sq->sqVA;
    auto tpId = (sq->tp_id) & ((1U << 24U) - 1U);
    auto rmtObjId = (sq->rmtObjId) & 0xFFFFF;
    auto tokenId = (sq->localTokenId) & 0xFFFFF;
	
    __gm__ uint32_t* wqeWords = (__gm__ uint32_t*)(sqBaseAddr + wqeSize * (myHead % qpDepth));

    uint32_t w[16];
    // Word 0: sqeBbIdx[15:0] | flag[23:16] | rsv0[26:24] | nf[27] | tokenEn[28] | rmtJettyType[30:29] | owner[31]
    auto owner = (myHead & qpDepth) == 0 ? 1 : 0;
    w[0] = (0u << 0)                    // sqeBbIdx = 0
                | (0b00100000u << 16)          // flag
                | (0u << 24)                    // rsv0
                | (0u << 27)                    // nf
                | (1u << 28)     				// tokenEn
                | (1u << 29)          			// rmtJettyType
                | ((uint32_t)owner << 31);      // owner

    // Word 1: targetHint[7:0] | opcode[15:8] | rsv1[21:16] | inlineMsgLen[31:22]
    w[1] = (0u << 0)
                | (static_cast<uint32_t>(URMAOPCODE::OP_WRITE) << 8)
                | (0u << 16)                    // rsv1
                | (0u << 22);                   // inlineMsgLen

    // Word 2: tpId[23:0] | sgeNum[31:24]
    w[2] = (tpId << 0)
                | (1u << 24);                   // sgeNum = 1

    // Word 3: rmtJettyOrSegId[19:0] | rsv2[31:20]
    w[3] = (rmtObjId << 0)
                | (0u << 20);                   // rsv2

    // Word 4-5: rmtEidL (uint64_t)
	auto rmtEid = (JETTY_POS uint32_t *)(sq->rmtEid);
    w[4] = rmtEid[0];
    w[5] = rmtEid[1];

    // Word 6-7: rmtEidH (uint64_t)
    w[6] = rmtEid[2];
    w[7] = rmtEid[3];

    // Word 8: rmtTokenValue
    w[8] = 0;

    // Word 9: udfType[7:0] | reduceDataType[11:8] | reduceOpcode[15:12] | rsv3[31:16]
    w[9] = 0;

    // Word 10: rmtAddrLOrTokenId
    w[10] = remoteAddr & 0xFFFFFFFF;

    // Word 11: rmtAddrHOrTokenValue
    w[11] = remoteAddr >> 32;
   
	// SGE
    w[12] = messageLen;
    w[13] = 0;
    w[14] = (uint32_t)(localAddr & 0xFFFFFFFF);
    w[15] = (uint32_t)(localAddr >> 32);
    
    #pragma unroll
    for(int i = 0; i < 16; i ++){
        // asc_stwt(&wqeWords[i], w[i]);
        wqeWords[i] = w[i];
    }

}

__simt_callee__ inline void simt_write_wqe(JETTY_POS HcclAiRMAWQ *sq, uint32_t myHead, uint64_t localAddr, uint64_t remoteAddr, uint32_t messageLen, URMAOPCODE opcode)
{
    constexpr uint32_t shift = 14;
    constexpr uint32_t wqeSize = 64;
    constexpr uint32_t qpDepth = 1 << shift;

    // auto shift = 14;
    // auto wqeSize = sq->wqeSize;
    // auto qpDepth = sq->sqDepth;

    auto sqBaseAddr = sq->sqVA;
    auto tpId = (sq->tp_id) & ((1U << 24U) - 1U);
    auto rmtObjId = (sq->rmtObjId) & 0xFFFFF;
    auto tokenId = (sq->localTokenId) & 0xFFFFF;
    
#if 0
    // if (threadIdx.x == 0) printf("writing wqes to UB\n");
    __ubuf__ uint8_t* wqeAddr = (__ubuf__ uint8_t*)(get_imm(WQE_OUT_OFFSET) + wqeSize * threadIdx.x);
#else
    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (myHead % qpDepth));
    // if (threadIdx.x == 0) printf("writing wqes to GM, addr %p\n", wqeAddr);
#endif
    uint32_t ownBit = ~((myHead >> shift) & 1U); // owner=~((jfs.pi >> jfs.sqe_bb_shift)&0x1)
    ownBit <<= 31U; // [31] owner_bit
    ownBit |= 1 << 21; // enable cqe
    ownBit |= 1 << 29; // rmt_jetty_type
    ownBit |= 1 << 28; // token_enable
    *(WQE_OUT_POS uint32_t*)(wqeAddr) = ownBit; // [3:0]
    // *(WQE_OUT_POS uint32_t*)(wqeAddr + 4) = (uint32_t)opcode << 8U; // opcode=0x3, write
    *(WQE_OUT_POS uint32_t*)(wqeAddr + 4) = (uint32_t)URMAOPCODE::OP_WRITE << 8U; // opcode=0x3, write
    *(WQE_OUT_POS uint32_t*)(wqeAddr + 8) = (1U << 24U) | tpId;  // [31:24] num_sge = 1 [23:0] tp_id
    *(WQE_OUT_POS uint32_t*)(wqeAddr + 12) = rmtObjId; // [19:0] rmt_jetty_or_seg_id

    #pragma unroll
    for (uint32_t index = 0; index < 16; index++) {
        *(WQE_OUT_POS uint8_t*)(wqeAddr + 16 + index) = sq->rmtEid[index];
    }
    *(WQE_OUT_POS uint64_t*)(wqeAddr + 40) = (uint64_t)remoteAddr; // rmt_addr
    *(WQE_OUT_POS uint32_t*)(wqeAddr + 48) = (uint32_t)messageLen; // [31:0] length
    // *(WQE_OUT_POS uint32_t*)(wqeAddr + 48) = (uint32_t)5120; // [31:0] length

    *(WQE_OUT_POS uint64_t*)(wqeAddr + 56) = (uint64_t)localAddr;
}

template<int port_num = 1, int jetty_num = 1>
__simt_vf__ __aicore__ LAUNCH_BOUND(WQE_THREADS_NUM)
inline void simt_nw_mj_vf(int rankNum, __ubuf__ wqe_desc_array_t *wqe_desc_arr, uint32_t wqe_count, JETTY_POS HcclAiRMAWQ *sqs,
                __ubuf__ uint32_t* wqeSqIdx,
                __ubuf__ uint32_t* dstRankWqeCount,
                __ubuf__ uint32_t* sqWqeCount,
                __ubuf__ uint32_t* sqBaseHead,
                __ubuf__ uint32_t* sqCursor,
                __ubuf__ uint32_t* HeadRecordUB,
                __gm__ uint32_t* HeadRecordGM,
                uint32_t maxJettyNumOnUB,
                uint32_t maxHeadRecordSlot,
                uint32_t startRankId,
                uint32_t writerId)
{
    int jetty_per_rank = port_num * jetty_num;
    int total_jetty = rankNum * port_num * jetty_num;
    int tid = threadIdx.x;
    uint32_t warpId = tid >> 5U;
    uint32_t laneId = tid & 31U;

    uint32_t N32_num_per_jetty = maxHeadRecordSlot >> 5U;
    // auto DB_ARR = (__ubuf__ uint64_t *) get_imm(DB_ARRAY_OFFSET);
    // auto HEAD_ARR = (__ubuf__ uint32_t *) get_imm(HEAD_ARRAY_OFFSET);
    // auto PREV_HEAD_ARR = (__ubuf__ uint32_t *) get_imm(PREV_HEAD_ARRAY_OFFSET);
    // auto OUT_JETTY_IDX = (__ubuf__ uint32_t *) get_imm(OUT_JETTY_IDX_OFFSET);
    // auto PORT_IDX = (__ubuf__ uint32_t *) get_imm(PORT_IDX_OFFSET);
    // auto JETTY_IDX = (__ubuf__ uint32_t *) get_imm(JETTY_IDX_OFFSET);

    for (int idx = tid; idx < rankNum; idx += WQE_THREADS_NUM) {
        dstRankWqeCount[idx] = 0U;
    }
    for (int sqIdx = tid; sqIdx < total_jetty; sqIdx += WQE_THREADS_NUM) {
        sqWqeCount[sqIdx] = 0U;
        sqCursor[sqIdx] = 0U;
        sqBaseHead[sqIdx] = 0U;
    }
    for (uint32_t slotIdx = tid; slotIdx < maxJettyNumOnUB * N32_num_per_jetty; slotIdx += WQE_THREADS_NUM) {
        HeadRecordUB[slotIdx] = 0U;
    }
    __syncthreads();

    // Phase 1: Route every WQE to one SQ and count this batch per SQ.
    for (auto my_idx = tid; my_idx < wqe_count; my_idx += WQE_THREADS_NUM) {
        uint32_t my_dst_rank = wqe_desc_arr->dstRank[my_idx];

        // dstRankWqeOrdinal is this WQE's zero-based position among WQEs
        // targeting the same rank in the current writer batch.
        uint32_t dstRankWqeOrdinal = atomicAdd(dstRankWqeCount + my_dst_rank, 1U);
        uint32_t localSqIdx = (writerId + dstRankWqeOrdinal) % jetty_per_rank;
        uint32_t mySqIdx = my_dst_rank * jetty_per_rank + localSqIdx;
        wqeSqIdx[my_idx] = mySqIdx;
        atomicAdd(sqWqeCount + mySqIdx, 1U);
    }
    __syncthreads();

    // Phase 2: Reserve one contiguous head range per active SQ. This reduces
    // GM atomicAdd operations from one per WQE to one per active SQ per batch.
    for (int sqIdx = tid; sqIdx < total_jetty; sqIdx += WQE_THREADS_NUM) {
        uint32_t count = sqWqeCount[sqIdx];
        if (count > 0U) {
            auto mySq = &sqs[sqIdx];
            sqBaseHead[sqIdx] = atomicAdd((__gm__ uint32_t *)(mySq->headAddr), count);
        }
    }
    __syncthreads();

    // Phase 3: Assign each WQE an offset in its reserved SQ range and write it.
    for (auto my_idx = tid; my_idx < wqe_count; my_idx += WQE_THREADS_NUM) {
        uint64_t my_local_addr, my_remote_addr;
        uint32_t my_message_len, my_dst_rank;
        URMAOPCODE my_op;
        get_wqe_content(wqe_desc_arr, my_idx, my_local_addr, my_remote_addr, my_message_len, my_dst_rank, my_op);

        uint32_t mySqIdx = wqeSqIdx[my_idx];
        uint32_t localOffset = atomicAdd(sqCursor + mySqIdx, 1U);
        uint32_t my_head = sqBaseHead[mySqIdx] + localOffset;
        auto my_sq = &sqs[mySqIdx];
        simt_write_wqe_v4(my_sq, my_head, my_local_addr, my_remote_addr, my_message_len, my_op);
        asc_threadfence();
        uint32_t headRecordSlot = my_head % maxHeadRecordSlot;
        uint32_t wordSlot = headRecordSlot >> 5U;
        uint32_t jettyIdxOnUb = mySqIdx - startRankId * jetty_per_rank;
        uint32_t bitInWord = headRecordSlot & 31U;
        atomicOr(&HeadRecordUB[jettyIdxOnUb * N32_num_per_jetty + wordSlot], 1U << bitInWord);
    }

    __syncthreads();

    for (uint32_t sqIdx = warpId; sqIdx < total_jetty; sqIdx += WQE_THREADS_NUM) {
        uint32_t count = sqWqeCount[sqIdx];
        if (count > 0U) {
            uint32_t jettyIdxOnUb = sqIdx - startRankId * jetty_per_rank;
            for (uint32_t wordSlot = laneId; wordSlot < N32_num_per_jetty; wordSlot += WARP_SIZE) {
                uint32_t ubWordSlot = jettyIdxOnUb * N32_num_per_jetty + wordSlot;
                uint32_t mask = HeadRecordUB[ubWordSlot];
                if (mask != 0U) {
                    uint32_t gmWordSlot = sqIdx * (N32_num_per_jetty + UB_ALIGN_DATA_COUNT) + wordSlot;
                    atomicOr(HeadRecordGM + gmWordSlot, mask);
                }
            }
        }
    }

    // Phase 2: Each jetty is assigned to a thread. 
    // Since SIMT thread can not ring doorbell correctly, activated jetties' doorbell addr and head are written to UB parallelly. And atomicAdd is used to assign order.
    // Later, scalar will read them and ring doorbell sequentially.
//     for (auto my_idx = tid; my_idx < total_jetty; my_idx += WQE_THREADS_NUM)
//     // if (tid < total_jetty) 
//     {
//         // uint32_t prevHead = PREV_HEAD_ARR[my_idx];
//         auto my_sq = &sqs[my_idx];

//         if (my_sq->headAddr == 0) {
//             continue;
//         }
// #if 0
//         auto curHead = my_sq->localTokenId;
// #else
//         auto curHead = *(__gm__ uint32_t *)(my_sq->headAddr);
// #endif
//         // printf("[kfm] my_idx %d curHead %d\n", my_idx, curHead);
//         DB_ARR[my_idx] = my_sq->dbAddr;
//         HEAD_ARR[my_idx] = curHead % my_sq->sqDepth; // my_sq->headAddr
//         // if (curHead != prevHead) {
//         //     auto my_jetty = atomicAdd(OUT_JETTY_IDX, 1);
//         //     DB_ARR[my_jetty] = my_sq->dbAddr;
//         //     HEAD_ARR[my_jetty] = curHead; // my_sq->headAddr
//         //     PREV_HEAD_ARR[my_idx] = curHead;
//         // }
//     }
}


template<int port_num = 1, int jetty_num = 1>
__aicore__ inline void simt_nw_mj(int rankNum, __ubuf__ wqe_desc_array_t *wqe_desc_arr, uint32_t wqe_count, JETTY_POS HcclAiRMAWQ *sqs,
        __ubuf__ uint32_t* wqeSqIdx,
        __ubuf__ uint32_t* dstRankWqeCount,
        __ubuf__ uint32_t* sqWqeCount,
        __ubuf__ uint32_t* sqBaseHead,
        __ubuf__ uint32_t* sqCursor,
        __ubuf__ uint32_t* HeadRecordUB,
        __gm__ uint32_t* HeadRecordGM,
        uint32_t maxJettyNumOnUB,
        uint32_t maxHeadRecordSlot,
        uint32_t startRankId,
        uint32_t writerId)
{

    
    AscendC::Simt::VF_CALL<simt_nw_mj_vf<port_num, jetty_num>>(
        AscendC::Simt::Dim3{WQE_THREADS_NUM, 1, 1}, rankNum, wqe_desc_arr, wqe_count, sqs,
            wqeSqIdx,
            dstRankWqeCount,
            sqWqeCount,
            sqBaseHead,
            sqCursor,
            HeadRecordUB,
            HeadRecordGM,
            maxJettyNumOnUB,
            maxHeadRecordSlot,
            startRankId,
            writerId);

    set_flag(PIPE_V, PIPE_S, (event_t) 0);
    wait_flag(PIPE_V, PIPE_S, (event_t) 0);

}


__aicore__ inline void cacheWriteThrough(__gm__ uint8_t* sourceAddr, uint64_t length)
{
    __gm__ uint8_t* start = (__gm__ uint8_t*)((uint64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t* end = (__gm__ uint8_t*)(((uint64_t)sourceAddr + length) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
        asm volatile("");
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[i]);
        asm volatile("");
    }
}

__aicore__ inline void URMAPollCQ(__gm__ AiURMAWQ *sq, __gm__ AiURMACQ *cq)
{
    uint32_t idx = *(__gm__ uint32_t*)(sq->headAddr);
    if (idx == 0) {
        return;
    }
    // AscendC::printf("pollcq: sqDB %p, idx %d\n", sq->dbAddr, idx);

    auto cqBaseAddr = cq->cqVA;
    auto cqeSize = cq->cqeSize;
    auto depth = cq->cqDepth;
    uint32_t curTail = *(__gm__ uint32_t*)(cq->tailAddr);
    while (curTail != idx) {
        __gm__ uint32_t* cqeAddr = (__gm__ uint32_t*)(cqBaseAddr + cqeSize * (curTail & (depth - 1)));
        uint32_t cqeByte4 = *(__gm__ uint32_t*)cqeAddr;
        int64_t start = AscendC::GetSystemCycle(); // reserved for timeout check
        while ((cqeByte4 & (1 << 2)) == 0) {
            int64_t tmp = AscendC::GetSystemCycle(); // reserved for timeout check
            if (tmp - start > 3000000) {
                AscendC::printf("pollcq timeout: sqDB %p, idx %d\n", sq->dbAddr, idx);
                break;
            }
            cacheWriteThrough((__gm__ uint8_t*)cqeAddr, 8);
            cqeByte4 = *(__gm__ uint32_t*)cqeAddr;
        }
        // Check CQE status
        uint32_t status = cqeByte4 & 0xFF000000;    status >>= 24;
        uint32_t subStatus = cqeByte4 & 0x00FF0000; subStatus >>= 16;
        if (status != 0 || subStatus != 0) {
            AscendC::printf("status=%d != 0 || subStatus=%d != 0\n", (int)status, (int)subStatus);
        }
        curTail++;
    }
    uint64_t curJFCDB = curTail & 0x3FFFFF;
    st_dev(curJFCDB, reinterpret_cast<__gm__ uint32_t*>(cq->dbAddr), 0);
    // Update JFS tail
    auto curHardwareTail = sq->tailAddr;
    *(__gm__ uint32_t*)curHardwareTail = curTail;
}

#define OFFSET_OF(X, Y, Z, x, y, z)  ((x) * (Y) * (Z) + (y) * (Z) + (z))
__aicore__ inline void URMAPollCQ(uint32_t pe, uint32_t portNum, uint32_t qpNum, __gm__ AiURMAWQ *sqs, __gm__ AiURMACQ *cq) {
    for (int portIdx = 0; portIdx < portNum; portIdx++) {
        for (int qpIdx = 0; qpIdx < qpNum; qpIdx++) {
            auto offset = OFFSET_OF(0, portNum, qpNum, pe, portIdx, qpIdx);
            URMAPollCQ(sqs + offset, cq + offset);
        }
    }

    // for (int portIdx = FULLMESH_NUM; portIdx < PORT_NUM; portIdx++) {
    //     for (int qpIdx = 0; qpIdx < qpNum; qpIdx++) {
    //         auto offset = OFFSET_OF(0, portNum, qpNum, pe, portIdx, qpIdx);
    //         URMAPollCQ(sqs + offset, cq + offset);
    //     }
    // }
}


template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::URMASendToken()
{
    uint64_t timepoint[10];
    // timepoint[0] = AscendC::GetSystemCycle();
    uint32_t sendExpNum, startExpId, endExpId;
    TBuf<> expertCntListBuf, expertOffsetListBuf, expertOrderedOffsetListBuf,
        remoteAddrListBuf, finishedWqeDescNumBuf, totalSendNumBuf, headReadyBuf, headRecordBuf;
    const uint32_t selfExpertBegin = epRankId_ * moeExpertNumPerRank_;
    const uint32_t selfExpertEnd = selfExpertBegin + moeExpertNumPerRank_;
    const uint32_t expertNumBeforeSelf = selfExpertBegin;
    const uint32_t expertNumAfterSelf = moeExpertNum_ - selfExpertEnd;

    if (aivUsedRemoteWqe_ == 1U && expertNumBeforeSelf > 0U && expertNumAfterSelf > 0U) {
        // A single WQE core cannot own two disjoint ranges. Keep the original
        // full range as a small-core fallback; buildRemoteWqeDesc filters self.
        startExpId = 0U;
        sendExpNum = moeExpertNum_;
        endExpId = moeExpertNum_;
    } else {
        uint32_t beforeSelfAivNum = 0U;
        if (expertNumBeforeSelf == 0U) {
            beforeSelfAivNum = 0U;
        } else if (expertNumAfterSelf == 0U) {
            beforeSelfAivNum = aivUsedRemoteWqe_;
        } else {
            beforeSelfAivNum = static_cast<uint32_t>(
                static_cast<uint64_t>(aivUsedRemoteWqe_) * expertNumBeforeSelf /
                (expertNumBeforeSelf + expertNumAfterSelf));

            uint32_t minBeforeSelfAivNum =
                aivUsedRemoteWqe_ > expertNumAfterSelf ?
                aivUsedRemoteWqe_ - expertNumAfterSelf : 1U;
            uint32_t maxBeforeSelfAivNum =
                aivUsedRemoteWqe_ < expertNumBeforeSelf ?
                aivUsedRemoteWqe_ : expertNumBeforeSelf;
            if (maxBeforeSelfAivNum >= aivUsedRemoteWqe_) {
                maxBeforeSelfAivNum = aivUsedRemoteWqe_ - 1U;
            }
            if (beforeSelfAivNum < minBeforeSelfAivNum) {
                beforeSelfAivNum = minBeforeSelfAivNum;
            }
            if (beforeSelfAivNum > maxBeforeSelfAivNum) {
                beforeSelfAivNum = maxBeforeSelfAivNum;
            }
        }

        uint32_t workerId = aivId_;
        uint32_t rangeExpertNum = expertNumBeforeSelf;
        uint32_t rangeAivNum = beforeSelfAivNum;
        uint32_t rangeBeginExpertId = 0U;
        if (aivId_ >= beforeSelfAivNum) {
            workerId = aivId_ - beforeSelfAivNum;
            rangeExpertNum = expertNumAfterSelf;
            rangeAivNum = aivUsedRemoteWqe_ - beforeSelfAivNum;
            rangeBeginExpertId = selfExpertEnd;
        }

        const uint32_t baseExpertNum = rangeExpertNum / rangeAivNum;
        const uint32_t remainderExpertNum = rangeExpertNum % rangeAivNum;
        sendExpNum = baseExpertNum + (workerId < remainderExpertNum ? 1U : 0U);
        const uint32_t expertOffset = workerId * baseExpertNum +
            (workerId < remainderExpertNum ? workerId : remainderExpertNum);
        startExpId = rangeBeginExpertId + expertOffset;
        endExpId = startExpId + sendExpNum;
    }
    // printf("URMASendToken Info aiv %d, total %d, compareTokenOriginalOffset %d, e %d, n %d\n",
    //         aivId_, epWorldSize_, startRankId, endRankId, sendRankNum);
    // AscendC::printf("[kfm] rankId %d, aivId %d, startRankId %d, endRankId %d, sendRankNum %d",
        // epRankId_, aivId_, startRankId, endRankId, sendRankNum);
    if (sendExpNum==0)
        return;

    uint32_t startRankId = startExpId / moeExpertNumPerRank_;
    uint32_t endRankId = endExpId / moeExpertNumPerRank_;
    uint32_t sendRankNum = (endExpId % moeExpertNumPerRank_ == 0) ? (endRankId - startRankId) : (endRankId - startRankId + 1);

    uint32_t N32numPerJetty = Ceil(axisBS_ * axisK_, 32U);
    uint32_t maxHeadRecordSlot = N32numPerJetty * 32U;
    uint32_t maxJettyNumOnUB = (sendExpNum / moeExpertNumPerRank_) + 2U; // +1是因为可能跨相邻两卡专家边界，再+1是可能跨本卡
    // AscendC::printf("====== kfm1 ======\n");
    tpipe_->InitBuffer(expertIdsBuf_, Ceil(expertIdsCnt_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(expertCntListBuf, sendExpNum * sizeof(uint32_t));
    tpipe_->InitBuffer(expertOffsetListBuf, sendExpNum * axisBS_ * sizeof(uint32_t));
    tpipe_->InitBuffer(expertOrderedOffsetListBuf, sendExpNum * axisBS_ * sizeof(uint32_t));
    tpipe_->InitBuffer(totalSendNumBuf, UB_ALIGN);
    tpipe_->InitBuffer(finishedWqeDescNumBuf, UB_ALIGN);
    tpipe_->InitBuffer(remoteAddrListBuf, sizeof(uint64_t) * sendRankNum);
    tpipe_->InitBuffer(headReadyBuf, UB_ALIGN);
    tpipe_->InitBuffer(headRecordBuf,
        Ceil(maxJettyNumOnUB * N32numPerJetty * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    

    validExpertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    LocalTensor<uint32_t> expertCntListLT = expertCntListBuf.Get<uint32_t>();
    LocalTensor<int32_t> expertOffsetListLT = expertOffsetListBuf.Get<int32_t>();
    LocalTensor<int32_t> expertOrderedOffsetListLT = expertOrderedOffsetListBuf.Get<int32_t>();
    LocalTensor<uint32_t> totalSendNumLT = totalSendNumBuf.Get<uint32_t>();
    LocalTensor<uint32_t> finishedWqeDescNumLT = finishedWqeDescNumBuf.Get<uint32_t>();
    LocalTensor<uint64_t> remoteAddrListLT = remoteAddrListBuf.Get<uint64_t>();
    LocalTensor<uint32_t> headReadyLT = headReadyBuf.Get<uint32_t>();
    LocalTensor<uint32_t> headRecordLT = headRecordBuf.Get<uint32_t>();


    TBuf<> tokenSendWqeDescBuf, wqeSqIdxBuf, dstRankWqeCountBuf,
        sqWqeCountBuf, sqBaseHeadBuf, sqCursorBuf;
    tpipe_->InitBuffer(wqeSqIdxBuf,
        Ceil(MAX_WQE_CNT * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(dstRankWqeCountBuf,
        Ceil(epWorldSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(sqWqeCountBuf,
        Ceil(totalJettyNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(sqBaseHeadBuf,
        Ceil(totalJettyNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(sqCursorBuf,
        Ceil(totalJettyNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);

    LocalTensor<uint32_t> wqeSqIdxLT = wqeSqIdxBuf.Get<uint32_t>();
    LocalTensor<uint32_t> dstRankWqeCountLT = dstRankWqeCountBuf.Get<uint32_t>();
    LocalTensor<uint32_t> sqWqeCountLT = sqWqeCountBuf.Get<uint32_t>();
    LocalTensor<uint32_t> sqBaseHeadLT = sqBaseHeadBuf.Get<uint32_t>();
    LocalTensor<uint32_t> sqCursorLT = sqCursorBuf.Get<uint32_t>();

    GM_ADDR sqsGM = GetSqsAddr();
    GM_ADDR cqsGM = sqsGM + sizeof(HcclAiRMAWQ) * totalJettyNum;

    // auto sqs = reinterpret_cast<__gm__ HcclAiRMAWQ *>(sqsGM);

    // for (uint32_t i = 0; i < totalJettyNum; ++i) {
    //     uint32_t currentHead = 0;

    //     if (sqs[i].headAddr != 0) {
    //         currentHead =
    //             *reinterpret_cast<__gm__ uint32_t *>(sqs[i].headAddr) % sqs[i].sqDepth;
    //     }

    //     PrevHeadArr.SetValue(i, currentHead);
    // }

    // PipeBarrier<PIPE_ALL>();

    SyncFunc<AscendC::HardEvent::V_S>();
    PipeBarrier<PIPE_ALL>();

    tpipe_->InitBuffer(tokenSendWqeDescBuf, Ceil(sizeof(wqe_desc_array_t), UB_ALIGN) * UB_ALIGN);
    LocalTensor<uint32_t> tokenSendWqeDescU32 = tokenSendWqeDescBuf.Get<uint32_t>();
    __ubuf__ wqe_desc_array_t* tokenSendWqeDesc = reinterpret_cast<__ubuf__ wqe_desc_array_t*>(tokenSendWqeDescU32.GetPhyAddr());

    // AscendC::printf("====== kfm2 ======\n");

    // AscendC::printf("====== kfm2.5 ======\n");
    // get_sqs_and_cqs_xb(sqsGM, cqsGM);
    // cacheWriteThrough(sqsGM, sizeof(HcclAiRMAWQ) * totalJettyNum);
    // cacheWriteThrough(cqsGM, sizeof(HcclAiRMACQ) * totalJettyNum);
    
    GM_ADDR headRecordGM = GetHeadRecordAddr();

    // // get_sqs_and_cqs_xb_ub_v3(reinterpret_cast<uint64_t>(sqsLT.GetPhyAddr()), 
    // //         reinterpret_cast<uint64_t>(cqsLT.GetPhyAddr()));
    // AscendC::printf("get_sqs_and_cqs_xb ok\n");
    // for (int i = 0; i < epWorldSize_; i++) {
    //     for (int j = 0; j < PORT_NUM; j++) {
    //         for (int k = 0; k < JETTY_NUM; k++) {
    //             auto sq = reinterpret_cast<__gm__ AiURMAWQ *>(sqsGM) + OFFSET_OF(epWorldSize_, PORT_NUM, JETTY_NUM, i, j, k);
    //             auto cq = reinterpret_cast<__gm__ AiURMACQ *>(cqsGM) + OFFSET_OF(epWorldSize_, PORT_NUM, JETTY_NUM, i, j, k);
    //             AscendC::printf("rank %d, port %d, jetty %d, sqVa %p, sqDB %p, sqDepth %d, headAddr %p, \
    //                 wqeSize %d, tpid %d, rmtEidL %p, rmtEidH %p, rmtObjId %d, rmtTokenValue %d\n", 
    //                 i, j, k, sq->sqVA, sq->dbAddr, sq->sqDepth, sq->headAddr, sq->wqeSize, sq->tp_id, 
    //                 (__gm__ uint64_t *)(sq->rmtEid), (__gm__ uint64_t *)(sq->rmtEid + 8), 
    //                 sq->rmtObjId, sq->rmtTokenValue);
    //             // AscendC::printf("rank %d, port %d, jetty %d, sqVa %p, localTokenId %d\n", i, j, k, sq->sqVA, sq->localTokenId);
    //         }
    //     }
    // }
    PipeBarrier<PIPE_ALL>();
    
    // AscendC::printf("====== kfm3 ======\n");
    totalSendNumLT(0) = 0;
    Duplicate(expertCntListLT, 0U, sendExpNum);

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    
    PipeBarrier<PIPE_ALL>();
    
    // timepoint[1] = AscendC::GetSystemCycle();

    // for(uint32_t i = 0; i < expertIdsCnt_; i++){
    //     int32_t dstExpId = validExpertIdsTensor_(i);    //专家列表值
    //     if(dstExpId >= beginExpId && dstExpId < endExpId){
    //         int32_t logicExpId = dstExpId - beginExpId;
    //         uint32_t preInfoNum = expertCntListLT(logicExpId);
    //         expertCntListLT(logicExpId) = preInfoNum + 1;
    //         expertOffsetListLT(logicExpId * axisBS_ + preInfoNum) = i;
    //     }
    // }
    // AscendC::printf("====== kfm4 ======\n");

    // for(uint32_t i = 0; i < expertIdsCnt_; i++){
    //     AscendC::printf("expertIdsCnt_[%d]: %d\n", i, validExpertIdsTensor_(i));
    // }
    
    PipeBarrier<PIPE_ALL>();
    AscendC::Simt::VF_CALL<simt_prepare_mapping>(
        AscendC::Simt::Dim3{PREPARE_THREADS_NUM, 1, 1},
        reinterpret_cast<__ubuf__ int32_t*>(validExpertIdsTensor_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(expertCntListLT.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(expertOffsetListLT.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(expertOrderedOffsetListLT.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(totalSendNumLT.GetPhyAddr()), // 发送给远端卡的token数量
        axisBS_,
        axisK_,
        startExpId,
        sendExpNum,
        moeExpertNumPerRank_,
        epRankId_);
    SyncFunc<AscendC::HardEvent::V_S>();
    PipeBarrier<PIPE_ALL>();

    // for(uint32_t i = 0; i < sendRankNum * moeExpertNumPerRank_ * axisBS_; i++){
    //     AscendC::printf("[kfm] aivId_[%d] expertOrderedOffsetListLT[%d]: %d\n", aivId_, i, expertOrderedOffsetListLT(i));
    // }
    
    // AscendC::printf("[kfm] aivId_[%d] totalSendNumLT: %d\n", aivId_, totalSendNumLT(0));

    // timepoint[2] = AscendC::GetSystemCycle();
    // AscendC::printf("====== kfm5 ======\n");
    DebugClock(3);
    for(uint32_t i = 0; i < sendRankNum; i++){
        remoteAddrListLT(i) = uint64_t(GetWindAddrByRankId(startRankId + i)) + expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_);
        // AscendC::printf("[sendBase] rank %d aiv %d dstRank %d base %p remoteBase %p\n",
        //     epRankId_, aivId_, startRankId + i,
        //     GetWindAddrByRankId(startRankId + i), remoteAddrListLT(i));
    }
    PipeBarrier<PIPE_ALL>();


    uint64_t srcBaseAddr = uint64_t(GetSendBufferAddrByTokenId(0));
    // uint32_t targetRemoteSendNum = 0;
    // for (uint32_t targetLocalExp = 0; targetLocalExp < sendExpNum; ++targetLocalExp) {
    //     uint32_t targetExpId = startExpId + targetLocalExp;
    //     if (targetExpId / moeExpertNumPerRank_ != epRankId_) {
    //         targetRemoteSendNum += expertCntListLT(targetLocalExp);
    //     }
    // }

    
    // AscendC::printf("====== kfm6 ======\n");

    GM_ADDR headReadyGM = GetHeadReadyAddr();
    GlobalTensor<uint32_t> headReadyGT;
    uint32_t localAivId = aivId_;
    GM_ADDR currHeadReadyGM = headReadyGM + static_cast<uint64_t>(localAivId) * HEAD_READY_STRIDE;
    headReadyGT.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(currHeadReadyGM));
    while (true) {
        DataCopy(headReadyLT, headReadyGT, UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        if (headReadyLT(0) == headReadyFlag) break;
    }
    headReadyLT.SetValue(0, 0U);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(headReadyGT, headReadyLT, UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    DebugClock(4);
    uint32_t loopIndex = 0;
    uint32_t finishedTokenNum = 0;
    uint32_t totalRemoteSendNum = totalSendNumLT(0);
    finishedWqeDescNumLT(0) = 0;
    constexpr uint32_t WARP_NUM = BUILD_THREADS_NUM / WARP_SIZE;
    uint32_t chunkNumPerExpert = Ceil(axisBS_, WARP_SIZE);
    uint32_t totalBlockNum = sendExpNum * chunkNumPerExpert;
    uint32_t totalLoopNum = Ceil(totalBlockNum, WARP_NUM);
    SyncFunc<AscendC::HardEvent::S_V>();
    // AscendC::printf("[kfm] totalRemoteSendNum %d\n", totalRemoteSendNum);

    
    while (finishedTokenNum < totalRemoteSendNum && loopIndex < totalLoopNum) {
        AscendC::Simt::VF_CALL<buildRemoteWqeDesc>(
            AscendC::Simt::Dim3{BUILD_THREADS_NUM, 1, 1},
            reinterpret_cast<__ubuf__ uint32_t*>(expertCntListLT.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(expertOrderedOffsetListLT.GetPhyAddr()),
            sendExpNum,
            startExpId,
            axisBS_,
            axisK_,
            srcBaseAddr,
            hCommuSize_,
            moeExpertNumPerRank_,
            epRankId_,
            reinterpret_cast<__ubuf__ uint64_t*>(remoteAddrListLT.GetPhyAddr()),
            static_cast<uint32_t>(URMAOPCODE::OP_WRITE),
            tokenSendWqeDesc->localAddr,
            tokenSendWqeDesc->remoteAddr,
            tokenSendWqeDesc->dstRank,
            tokenSendWqeDesc->messageLen,
            reinterpret_cast<__ubuf__ uint32_t*>(tokenSendWqeDesc->opcode),
            loopIndex,
            reinterpret_cast<__ubuf__ uint32_t*>(finishedWqeDescNumLT.GetPhyAddr())
        );
        SyncFunc<AscendC::HardEvent::V_S>();
        // AscendC::printf("====== kfm7 ======\n");

        // PipeBarrier<PIPE_ALL>();
        uint32_t finishedWqeDesc = finishedWqeDescNumLT(0);
        // AscendC::printf("[kfm] loopIndex %d finishedWqeDesc %d\n", loopIndex, finishedWqeDesc);
        if (finishedWqeDesc > 0) {
            tokenSendWqeDesc->size = finishedWqeDesc;
            PipeBarrier<PIPE_ALL>();
            simt_nw_mj<PORT_NUM, JETTY_NUM>(static_cast<int>(epWorldSize_), tokenSendWqeDesc, finishedWqeDesc,
                reinterpret_cast<__gm__ HcclAiRMAWQ*>(sqsGM),
                reinterpret_cast<__ubuf__ uint32_t*>(wqeSqIdxLT.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(dstRankWqeCountLT.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(sqWqeCountLT.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(sqBaseHeadLT.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(sqCursorLT.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(headRecordLT.GetPhyAddr()),
                reinterpret_cast<__gm__ uint32_t*>(headRecordGM),
                maxJettyNumOnUB,
                maxHeadRecordSlot,
                startRankId,
                aivId_
            );
            // for (uint32_t i = 0; i < totalJettyNum; i++) {
                //     uint32_t rankIdx = i / (PORT_NUM * JETTY_NUM);
                //     if (rankIdx < startRankId || rankIdx >= startRankId + sendRankNum) {
                //         continue;
                //     }
                //     // AscendC::printf("dbAddr %p, head %d\n", DbArr(i), HeadArr(i));
                //     uint32_t prevHead = PrevHeadArr(i);
                //     uint32_t currHead = HeadArr(i);
                //     if (currHead != prevHead) {
                //         st_dev(currHead, (__gm__ uint32_t *)(DbArr(i)), 0);
                //         PrevHeadArr(i) = currHead;
                //     }
                // }
                // OutJettyIdx(0) = 0;
            // PipeBarrier<PIPE_ALL>();
        }
        // DebugClock(5 + loopIndex);
        SyncFunc<AscendC::HardEvent::V_S>();
        
        finishedTokenNum += finishedWqeDesc;
        finishedWqeDescNumLT(0) = 0;
        loopIndex++;
        SyncFunc<AscendC::HardEvent::S_V>();
    }
    // AscendC::printf("[kfm] finish WQE write\n");

    DebugClock(5);
    // AscendC::printf(
    //     "[wqe-gen] rank %u aiv %u exp=[%u,%u) "
    //     "target=%u generated=%u loop=%u/%u\n",
    //     epRankId_, aivId_, startExpId, endExpId,
    //     totalRemoteSendNum, finishedTokenNum,
    //     loopIndex, totalLoopNum);
    // AscendC::printf("====== kfm8 ======\n");

    // for(int rid = startRankId; rid < startRankId + sendRankNum; rid ++) {
    //     if(rid == epRankId_) continue;
    //     aclshmemx_udma_quiet(static_cast<int>(rid));
    // }

    // AscendC::printf("====== kfm9 ======\n");

    // printf("URMASendToken time is %d, %d, %d, %d\n",    
    //         timepoint[1] - timepoint[0],
    //         timepoint[2] - timepoint[1],
    //         timepoint[3] - timepoint[2],
    //         timepoint[4] - timepoint[3]);
}


template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SelfCopyToken()
{
    // if (aivUsedSelfCopy_ == 0U) {
    //     return;
    // }

    const uint32_t selfCopyAivId = aivId_ - aivUsedRemoteWqe_;
    const uint32_t baseExpertNum = moeExpertNumPerRank_ / aivUsedSelfCopy_;
    const uint32_t remainderExpertNum = moeExpertNumPerRank_ % aivUsedSelfCopy_;
    const uint32_t sendExpNum = baseExpertNum + (selfCopyAivId < remainderExpertNum ? 1U : 0U);
    if (sendExpNum == 0U) {
        return;
    }

    const uint32_t localExpertOffset = selfCopyAivId * baseExpertNum +
        (selfCopyAivId < remainderExpertNum ? selfCopyAivId : remainderExpertNum);
    const uint32_t startExpId = epRankId_ * moeExpertNumPerRank_ + localExpertOffset;

    TBuf<> expertCntListBuf, expertOffsetListBuf, expertOrderedOffsetListBuf,
        totalSendNumBuf, sendTokenTempBuf, stageDoneBuf, stageDoneWorkBuf,
        stageDoneSumBuf;
    tpipe_->InitBuffer(expertIdsBuf_,
        Ceil(expertIdsCnt_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(expertCntListBuf,
        Ceil(sendExpNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(expertOffsetListBuf,
        Ceil(sendExpNum * axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(expertOrderedOffsetListBuf,
        Ceil(sendExpNum * axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(totalSendNumBuf, UB_ALIGN);
    tpipe_->InitBuffer(sendTokenTempBuf, hCommuSize_);
    tpipe_->InitBuffer(stageDoneBuf, GetActiveStageNum() * UB_ALIGN);
    tpipe_->InitBuffer(stageDoneWorkBuf, GetActiveStageNum() * UB_ALIGN);
    tpipe_->InitBuffer(stageDoneSumBuf, UB_ALIGN);

    validExpertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    LocalTensor<uint32_t> expertCntListLT = expertCntListBuf.Get<uint32_t>();
    LocalTensor<int32_t> expertOffsetListLT = expertOffsetListBuf.Get<int32_t>();
    LocalTensor<int32_t> expertOrderedOffsetListLT = expertOrderedOffsetListBuf.Get<int32_t>();
    LocalTensor<uint32_t> totalSendNumLT = totalSendNumBuf.Get<uint32_t>();
    LocalTensor<XOutType> sendTokenTempLT = sendTokenTempBuf.Get<XOutType>();
    LocalTensor<float> stageDoneLT = stageDoneBuf.Get<float>();
    LocalTensor<float> stageDoneWorkLT = stageDoneWorkBuf.Get<float>();
    LocalTensor<float> stageDoneSumLT = stageDoneSumBuf.Get<float>();

    totalSendNumLT.SetValue(0, 0U);
    Duplicate<uint32_t>(expertCntListLT, 0U, sendExpNum);

    DataCopyExtParams expertIdsCopyParams{
        1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsPadParams{false, 0U, 0U, 0U};
    DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_,
        expertIdsCopyParams, expertIdsPadParams);
    PipeBarrier<PIPE_ALL>();

    AscendC::Simt::VF_CALL<simt_prepare_mapping>(
        AscendC::Simt::Dim3{PREPARE_THREADS_NUM, 1, 1},
        reinterpret_cast<__ubuf__ int32_t*>(validExpertIdsTensor_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(expertCntListLT.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(expertOffsetListLT.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(expertOrderedOffsetListLT.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(totalSendNumLT.GetPhyAddr()),
        axisBS_,
        axisK_,
        startExpId,
        sendExpNum,
        moeExpertNumPerRank_,
        epRankId_);
    SyncFunc<AscendC::HardEvent::V_S>();
    DebugClock(3);

    // A self-copy core may consume a token staged by any stage core.
    WaitAllStageDone(stageDoneLT, stageDoneWorkLT, stageDoneSumLT);
    DebugClock(4);

    const uint64_t srcBaseAddr = uint64_t(GetSendBufferAddrByTokenId(0));
    const uint64_t selfWindowBaseAddr = uint64_t(GetWindAddrByRankId(epRankId_)) +
        expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_);
    GlobalTensor<XOutType> tokenSrcGMTensor;
    GlobalTensor<XOutType> tokenDstGMTensor;

    for (uint32_t localExpIdx = 0; localExpIdx < sendExpNum; ++localExpIdx) {
        const uint32_t targetExpId = startExpId + localExpIdx;
        const uint32_t targetExpertOnRank = targetExpId % moeExpertNumPerRank_;
        const uint32_t validOffsetCnt = expertCntListLT(localExpIdx);
        for (uint32_t targetExpOffset = 0; targetExpOffset < validOffsetCnt; ++targetExpOffset) {
            const int32_t offsetValue =
                expertOrderedOffsetListLT(localExpIdx * axisBS_ + targetExpOffset);
            const uint32_t tokenId = static_cast<uint32_t>(offsetValue) / axisK_;
            const uint64_t srcAddr = srcBaseAddr + tokenId * hCommuSize_;
            const uint64_t dstAddr = selfWindowBaseAddr +
                (targetExpertOnRank * axisBS_ + targetExpOffset) * hCommuSize_;

            tokenSrcGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XOutType*>(srcAddr));
            tokenDstGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XOutType*>(dstAddr));
            DataCopy(sendTokenTempLT, tokenSrcGMTensor,
                hCommuSize_ / sizeof(XOutType));
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopy(tokenDstGMTensor, sendTokenTempLT,
                hCommuSize_ / sizeof(XOutType));
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DebugClock(5);
}


template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::RingDoorbell()
{

    uint32_t HEAD_RECORD_BATCH = 128;
    uint32_t N32numPerJetty = Ceil(axisBS_ * axisK_, 32U);
    uint32_t maxHeadRecordSlot = N32numPerJetty * 32U;
    // uint32_t headRecordSize = maxHeadRecordSlot * UB_ALIGN;
    uint32_t activeStageNum = GetActiveStageNum();

    TBuf<> prevHeadBuf, headRecordBuf, rankTokenCntBuf, finishedTokenBuf, sqTailBuf, headReadyBuf,
        stageDoneBuf, stageDoneWorkBuf, stageDoneSumBuf, remoteRankNumBuf;
    tpipe_->InitBuffer(prevHeadBuf, Ceil(totalJettyNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(headRecordBuf,  UB_ALIGN);
    tpipe_->InitBuffer(rankTokenCntBuf, Ceil(epWorldSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(finishedTokenBuf, Ceil(epWorldSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(sqTailBuf, Ceil(totalJettyNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(headReadyBuf, Ceil(aivUsedRemoteWqe_, 8) * 8 * UB_ALIGN);
    tpipe_->InitBuffer(stageDoneBuf, activeStageNum * UB_ALIGN);
    tpipe_->InitBuffer(stageDoneWorkBuf, activeStageNum * UB_ALIGN);
    tpipe_->InitBuffer(stageDoneSumBuf, UB_ALIGN);
    tpipe_->InitBuffer(remoteRankNumBuf, UB_ALIGN);
    LocalTensor<uint32_t> prevHeadLT = prevHeadBuf.Get<uint32_t>();
    LocalTensor<uint32_t> headRecordLT = headRecordBuf.Get<uint32_t>();
    LocalTensor<uint32_t> rankTokenCntLT = rankTokenCntBuf.Get<uint32_t>();
    LocalTensor<uint32_t> finishedTokenLT = finishedTokenBuf.Get<uint32_t>();
    LocalTensor<uint32_t> sqTailLT = sqTailBuf.Get<uint32_t>();
    LocalTensor<uint32_t> headReadyLT = headReadyBuf.Get<uint32_t>();
    LocalTensor<float> stageDoneLT = stageDoneBuf.Get<float>();
    LocalTensor<float> stageDoneWorkLT = stageDoneWorkBuf.Get<float>();
    LocalTensor<float> stageDoneSumLT = stageDoneSumBuf.Get<float>();
    LocalTensor<uint32_t> remoteRankNumLT = remoteRankNumBuf.Get<uint32_t>();

    GM_ADDR sqsGM = GetSqsAddr();
    GM_ADDR cqsGM = sqsGM + sizeof(HcclAiRMAWQ) * totalJettyNum;
    GM_ADDR headReadyGM = GetHeadReadyAddr();
    GM_ADDR headRecordGM = GetHeadRecordAddr();
    __gm__ HcclAiRMAWQ* sqs = reinterpret_cast<__gm__ HcclAiRMAWQ *>(sqsGM);

    GlobalTensor<uint32_t> headReadyGT;
    headReadyGT.SetGlobalBuffer((__gm__ uint32_t*)headReadyGM);
    GlobalTensor<uint32_t> headRecordGT;
    headRecordGT.SetGlobalBuffer((__gm__ uint32_t*)headRecordGM);
// 准备sqs cqs
    get_sqs_and_cqs_xb(sqsGM, cqsGM);
    cacheWriteThrough(sqsGM, sizeof(HcclAiRMAWQ) * totalJettyNum);
    cacheWriteThrough(cqsGM, sizeof(HcclAiRMACQ) * totalJettyNum);

    DebugClock(1);

// 清理headRecord
    // TBuf<> cleanRecordBuf;
    // tpipe_->InitBuffer(cleanRecordBuf, headRecordSize);
    // LocalTensor<uint32_t> cleanRecordTensor = cleanRecordBuf.Get<uint32_t>();
    // Duplicate<uint32_t>(cleanRecordTensor, 0U, headRecordSize / sizeof(uint32_t));
    // SyncFunc<AscendC::HardEvent::V_MTE3>();

    // GlobalTensor<uint32_t> headRecordTensor;
    // DataCopyParams cleanRecordParams{static_cast<uint16_t>(maxHeadRecordSlot), 1, 0, static_cast<uint16_t>(WIN_ADDR_ALIGN / UB_ALIGN - 1)};
    // for (uint32_t jettyId = 0; jettyId < totalJettyNum; ++jettyId) {
    //     headRecordTensor.SetGlobalBuffer((__gm__ uint32_t*)(GetHeadRecordAddr() + jettyId * axisBS_ * axisK_ * WIN_ADDR_ALIGN));
    //     DataCopy(headRecordTensor, cleanRecordTensor, cleanRecordParams);
    // }
    // SyncFunc<AscendC::HardEvent::MTE3_S>();

    AscendC::Simt::VF_CALL<set_gm_value>(
        AscendC::Simt::Dim3{SET_GM_VALUE_THREAD_NUM, 1, 1},
        reinterpret_cast<__gm__ uint32_t*>(headRecordGM),
        totalJettyNum * (N32numPerJetty + UB_ALIGN_DATA_COUNT),
        sizeof(uint32_t),
        0U
    );
    SyncFunc<AscendC::HardEvent::V_S>();
    // PipeBarrier<PIPE_V>();
    DebugClock(2);

// 读取prevHead
    for (uint32_t i = 0; i < totalJettyNum; ++i) {
        uint32_t currentHead = 0;
        uint32_t rankId = i / (PORT_NUM * JETTY_NUM);
        if ((rankId != epRankId_) && (sqs[i].headAddr != 0)) {
            currentHead = *reinterpret_cast<__gm__ uint32_t *>(sqs[i].headAddr);
        }
        prevHeadLT.SetValue(i, currentHead);
    }
    DebugClock(3);

// 设置headReady flag
    // 已读取prevHead，设置flag通知WQE核可以修改Head值并写入WQE
    // DataCopyParams readyFlagCopyParams{
    //     static_cast<uint16_t>(aivUsedRemoteWqe_), 1, 0,
    //     static_cast<uint16_t>(HEAD_READY_STRIDE / UB_ALIGN - 1)};
    // Duplicate<uint32_t>(headReadyLT, headReadyFlag, SIZE_ALIGN_256 / sizeof(uint32_t), Ceil(aivUsedRemoteWqe_, 8), 1, 8); // repeatTime, dstBlockStride, dstRepeatStride
    // SyncFunc<AscendC::HardEvent::V_MTE3>();
    // DataCopy(headReadyGT, headReadyLT, readyFlagCopyParams);
    // SyncFunc<AscendC::HardEvent::MTE3_S>();
    AscendC::Simt::VF_CALL<set_gm_value>(
        AscendC::Simt::Dim3{128, 1, 1},
        reinterpret_cast<__gm__ uint32_t*>(headReadyGM),
        aivUsedRemoteWqe_,
        HEAD_READY_STRIDE,
        headReadyFlag
    );
    SyncFunc<AscendC::HardEvent::V_S>();

    DebugClock(4);

// 计算token发送总数
    // CalTokenSendExpertCnt depends on the same four UB buffers initialized by CalCumSum.
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(dstExpBuf_, maxSize_);
    tpipe_->InitBuffer(subExpBuf_, maxSize_);
    tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
    ExpIdsCopyAndMaskCal();

    // Duplicate<uint32_t>(rankTokenCntLT, 0U, epWorldSize_);
    Duplicate<uint32_t>(finishedTokenLT, 0U, epWorldSize_);
    Duplicate<uint32_t>(sqTailLT, 0U, totalJettyNum);
    // Duplicate<uint32_t>(remoteRankNumLT, 0U, UB_ALIGN_DATA_COUNT);
    // SyncFunc<AscendC::HardEvent::V_S>();
    // uint32_t ranksLeftToSend = 0;
    // uint32_t expertIdsCalCnt = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    // for (uint32_t remoteRank = 0; remoteRank < epWorldSize_; ++remoteRank) {
    //     uint32_t tokenCntTotal = 0;
    //     if (remoteRank == epRankId_) continue;
    //     for (uint32_t expIdx = 0; expIdx < moeExpertNumPerRank_; ++expIdx) {
    //         int32_t tokenCntCurrExp = 0;
    //         uint32_t expertId = remoteRank * moeExpertNumPerRank_ + expIdx;
    //         CalTokenSendExpertCnt(expertId, expertIdsCalCnt, tokenCntCurrExp);
    //         tokenCntTotal += tokenCntCurrExp;
    //     }
    //     rankTokenCntLT.SetValue(remoteRank, tokenCntTotal);
    //     if (tokenCntTotal > 0) {
    //         ranksLeftToSend++;
    //     }
    // }
    
    remoteRankNumLT(0) = 0;
    SyncFunc<AscendC::HardEvent::S_V>();
    AscendC::Simt::VF_CALL<calc_rank_token_cnt>(
        AscendC::Simt::Dim3{CALC_CNT_THREADS_NUM, 1, 1},
        reinterpret_cast<__ubuf__ int32_t*>(validExpertIdsTensor_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(rankTokenCntLT.GetPhyAddr()),
        isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_,
        moeExpertNumPerRank_,
        reinterpret_cast<__ubuf__ uint32_t*>(remoteRankNumLT.GetPhyAddr()),
        epRankId_,
        epWorldSize_
    );
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t ranksLeftToSend = remoteRankNumLT(0);

    DebugClock(5);

// 等待token本卡搬运到到位
    WaitAllStageDone(stageDoneLT, stageDoneWorkLT, stageDoneSumLT);

    uint32_t prevHead = 0;
    uint32_t currHead = 0;

// 轮询headRead敲门铃
    uint32_t loopIndex = 0;
    uint32_t max_batch_N32_num = UB_ALIGN_DATA_COUNT; // 一个batch拷贝一个UB_ALIGN包含的u32数量
    while (true) {
        for (uint32_t sqIdx = 0; sqIdx < totalJettyNum; ++sqIdx) {
            uint32_t rankId = sqIdx / (PORT_NUM * JETTY_NUM);
            if (rankId == epRankId_) continue;
            if (rankTokenCntLT(rankId) == 0 ||
                finishedTokenLT(rankId) >= rankTokenCntLT(rankId)) {
                continue;
            }
            prevHead = prevHeadLT(sqIdx) + sqTailLT(sqIdx);

            uint32_t wqeCount = 0;
            uint32_t remainingTokenCnt = rankTokenCntLT(rankId) - finishedTokenLT(rankId);
            uint32_t headRecordBase = sqIdx * (N32numPerJetty + UB_ALIGN_DATA_COUNT);
            
            while (wqeCount < maxHeadRecordSlot && wqeCount < remainingTokenCnt) {
                uint32_t prevHeadNew = prevHead + wqeCount;
                uint32_t recordSlot = prevHeadNew % maxHeadRecordSlot;
                uint32_t batchCount = max_batch_N32_num;
                uint32_t recordSlotN32Base = recordSlot >> 5U;
                uint32_t recPos = recordSlot & 31U;              // 本word内起始扫描位
                uint32_t untilWrap = N32numPerJetty - recordSlotN32Base; 
                batchCount = batchCount > untilWrap ? untilWrap : batchCount;
                uint32_t remainingInScan = remainingTokenCnt - wqeCount;
                uint32_t maxRemainingN32 = Ceil(remainingInScan + recPos, 32U); // 单卡多jetty不会有这么多
                batchCount = batchCount > maxRemainingN32 ? maxRemainingN32 : batchCount;
                uint32_t scanLimit = (batchCount << 5U) - recPos;
                scanLimit = scanLimit > remainingInScan ? remainingInScan : scanLimit;
                SyncFunc<AscendC::HardEvent::S_MTE2>();
                // DataCopyParams copyRecordParams{static_cast<uint16_t>(batchCount), 1, static_cast<uint16_t>(WIN_ADDR_ALIGN / UB_ALIGN - 1), 0};
                DataCopy(headRecordLT, headRecordGT[headRecordBase + recordSlotN32Base],  UB_ALIGN_DATA_COUNT); // 下一个U32
                SyncFunc<AscendC::HardEvent::MTE2_S>();

                int32_t readyCount = 0;
                for (uint32_t n32Idx = 0; n32Idx < batchCount; n32Idx++) {
                    uint32_t maskWord = headRecordLT(n32Idx);
                    if (n32Idx == 0 && recPos != 0U) {
                        // 只统计从 recPos 起的连续置位：右移后低位（已消费/未置位）不再参与，
                        // 高位补0保证 ScalarGetSFFValue<0> 必然找到终止位，返回值=从recPos起就绪个数。
                        uint32_t shifted = maskWord >> recPos;
                        int32_t cnt = ScalarGetSFFValue<0>(shifted);
                        readyCount += cnt;
                        if (cnt < static_cast<int32_t>(32U - recPos)) {
                            break;   // 半满/空：本word就绪数已统计完，停止
                        }
                        continue;   // 从recPos到word末尾全满：继续下一个word
                    }
                    int32_t firstInvalidIdx = ScalarGetSFFValue<0>(maskWord);
                    if (firstInvalidIdx == -1) { // 到了32个
                        readyCount += 32U;
                        continue;
                    } else {
                        readyCount += firstInvalidIdx;
                        break;
                    }
                }
                readyCount = readyCount > static_cast<int32_t>(scanLimit) ?
                    static_cast<int32_t>(scanLimit) : readyCount;
                wqeCount += readyCount;
                if (readyCount < static_cast<int32_t>(scanLimit)) break;
            }

            // DataCopyParams copyRecordParams{static_cast<uint16_t>(batchCount), 1, static_cast<uint16_t>(WIN_ADDR_ALIGN / UB_ALIGN - 1), 0};
            // DataCopy(headRecordLT, headRecordGT[headRecordBase + recordSlot * WIN_ADDR_ALIGN / sizeof(uint32_t)], 
            //     copyRecordParams);
            // SyncFunc<AscendC::HardEvent::MTE2_S>();

            // // uint32_t readyCount = 0;
            // for (uint32_t i = 0; i < batchCount; ++i) {
            //     uint32_t currExpectHead = prevHeadNew + i;
            //     if (headRecordLT(i * UB_ALIGN_DATA_COUNT) != currExpectHead + 1U) break;
            //     // readyCount++;
            //     wqeCount++;
            // }
            
            // wqeCount += readyCount;
            
            // if (readyCount < batchCount) break;
            // }
            sqTailLT(sqIdx) += wqeCount;

            if (wqeCount > 0) {
                // DebugClock(6 + loopIndex);
                // else if (loopIndex == 1) {
                //     DebugClock(7);
                // }
                // else if (loopIndex == 2) {
                //     DebugClock(8);
                // }
                // AscendC::printf(
                //     "[db] rank=%u dst=%u prev=%u publish=%u "
                //     "batch=%u finished=%u target=%u\n",
                //     epRankId_, rankId, prevHead,
                //     prevHead + sqTailLT(sqIdx),
                //     wqeCount,
                //     finishedTokenLT(rankId) + wqeCount,
                //     rankTokenCntLT(rankId));
                currHead = prevHead + wqeCount;
                st_dev(currHead, (__gm__ uint32_t *)(sqs[sqIdx].dbAddr), 0);
                loopIndex++;
            }

            finishedTokenLT(rankId) += wqeCount;
            if (finishedTokenLT(rankId) >= rankTokenCntLT(rankId)) {
                ranksLeftToSend--;
            }
        }
        if (ranksLeftToSend == 0) break;
    }
    // DebugClock(6 + loopIndex);
    DebugClock(6);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::Process()
{
    DebugClock(0);
    //  AscendC::printf("====== chy process ======\n");
    if ASCEND_IS_AIV {          // 全aiv处理
        // All stage flags are cleared in Init. Synchronize once before any
        // stage core publishes the completion flag for this launch.
        // SyncAll<true>();
        // tpipe_->Reset();
        // AscendC::printf("====== chy aiv process ======\n");
        // AscendC::printf("====== chy0 ======\n");
        // AscendC::printf("ismc2Context: %d\n", isMc2Context_);
        // GM_ADDR r0_ptr = (GM_ADDR)aclshmem_ptr((GM_ADDR)shmemBuffer_, 0);
        // GM_ADDR r1_ptr = (GM_ADDR)aclshmem_ptr((GM_ADDR)shmemBuffer_, 1);
        // AscendC::printf("[kfm] Addr check: r0 shmemBuffer %p, r1 shmemBuffer %p\n", r0_ptr, r1_ptr);

        if (aivId_ < aivUsedStage_) {
            // AscendC::printf("====== chy11 [aivId_ %d]======\n", aivId_);
            AllToAllDispatch(); // 前面核写本地 send buffer / shared expert
            // AscendC::printf("====== chy12 [aivId_ %d]======\n", aivId_);
            // PipeBarrier<PIPE_ALL>();
            DebugClock(1);
            // DebugClock(3);
            // SyncAll<true>();
            PublishStageDone();
            tpipe_->Reset();
            DebugClock(2);
            if (aivId_ < aivUsedRemoteWqe_) {
                URMASendToken();
            } else {
                SelfCopyToken();
            }
            DebugClock(6);
            // AscendC::printf("====== chy32 [aivId_ %d]======\n", aivId_);
        } else if (aivId_ < aivCumSumStart_) {
            RingDoorbell();
            // DebugClock(6);
            // DebugClock(8);
        } else {
            // AscendC::printf("====== chy21 [aivId_ %d]======\n", aivId_);
            CalCumSum();        // 后面核发送当前卡给每个专家的 tokenCnt，输出 epRecvCnt/exportTokenNums
            DebugClock(6);
            // DebugClock(7);
            // DebugClock(8);
            // AscendC::printf("====== chy22 [aivId_ %d]======\n", aivId_);
        }
        // localWindowCopy中包含reset操作，需确保前面操作完成
        // AscendC::printf("====== chy41 [aivId_ %d]======\n", aivId_);
        // SyncAll<true>();

        PipeBarrier<PIPE_ALL>();
        DebugClock(7);
        LocalWindowCopy();      // 本卡上专家数据连续化，输出expandX/scales/expandIdx
        PipeBarrier<PIPE_ALL>();
        DebugClock(8);

        // AscendC::printf("====== chy42 [aivId_ %d]======\n", aivId_);

#ifdef DEBUG_CLOCK_ON
        tpipe_->Reset();
        tpipe_->InitBuffer(clockResultBuf_, Ceil(sizeof(int32_t) * 16U, UB_ALIGN) * UB_ALIGN);
        clockResultLT = clockResultBuf_.Get<int32_t>();
        uint32_t timeOffset = aivId_;
        GlobalTensor<int32_t> clockResultOut;
        clockResultOut.SetGlobalBuffer((__gm__ int32_t*)(sendTpCountOutGM_));
        for (int i = 0; i < 15; i++) {
            int64_t timeCast = timePoint[i+1] - timePoint[i];
            clockResultLT(i) = static_cast<int32_t>(timeCast);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(clockResultOut[timeOffset * 16], clockResultLT, 16);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
#endif
    }
}

} // MoeDistributeDispatchV2FullMeshImpl
#endif // MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H
