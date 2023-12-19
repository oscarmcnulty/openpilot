package ai.flow.modeld;

import ai.flow.common.ParamsInterface;
import ai.flow.common.transformations.Camera;
import ai.flow.common.utils;
import ai.flow.definitions.Custom;
import ai.flow.definitions.Definitions;
import ai.flow.modeld.messages.MsgCameraOdometery;
import ai.flow.modeld.messages.MsgModelDataV2;
import messaging.ZMQPubHandler;
import messaging.ZMQSubHandler;
import org.capnproto.PrimitiveList;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Core;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static ai.flow.common.SystemUtils.getUseGPU;
import static ai.flow.common.utils.numElements;
import static ai.flow.sensor.messages.MsgFrameBuffer.updateImageBuffer;

public class ModelExecutorF2 extends ModelExecutor {

    public boolean stopped = true;
    public boolean initialized = false;
    public ParsedOutputs outs;
    public long timePerIt = 0;
    public long iterationNum = 1;

    public static int[] imgTensorShape = {1, 12, 128, 256};
    public static final int[] desireTensorShape = {1, 8};
    public static final int[] trafficTensorShape = {1, 2};
    public static final int[] stateTensorShape = {1, 512};
    public static final int[] outputTensorShape = {1, 6524}; // nuclear grade model output size

    public static final Map<String, int[]> inputShapeMap = new HashMap<>();
    public static final Map<String, int[]> outputShapeMap = new HashMap<>();
    public final INDArray desireNDArr = Nd4j.zeros(desireTensorShape);
    public final INDArray trafficNDArr = Nd4j.zeros(trafficTensorShape);
    public final INDArray stateNDArr = Nd4j.zeros(stateTensorShape);
    public final float[] netOutputs = new float[(int)numElements(outputTensorShape)];
    public final INDArray augmentRot = Nd4j.zeros(3);
    public final INDArray augmentTrans = Nd4j.zeros(3);
    public final float[]prevDesire = new float[CommonModelF2.DESIRE_LEN];
    public final float[]desireIn = new float[CommonModelF2.DESIRE_LEN];
    public final Map<String, INDArray> inputMap =  new HashMap<>();
    public final Map<String, float[]> outputMap =  new HashMap<>();
    public final Parser parser = new Parser();

    public final ParamsInterface params = ParamsInterface.getInstance();

    public static final int[] FULL_FRAME_SIZE = Camera.frameSize;
    public final ZMQPubHandler ph = new ZMQPubHandler();
    public final ZMQSubHandler sh = new ZMQSubHandler(true);
    public Definitions.LiveCalibrationData.Reader liveCalib;

    public long start, end, timestamp;
    public int lastFrameID = -1;
    public int firstFrameID = -1;
    public int totalFrameDrops = 0;
    public int frameDrops = 0; // per iteration
    public ModelRunner modelRunner;
    public MsgCameraOdometery msgCameraOdometery = new MsgCameraOdometery();
    public MsgModelDataV2 msgModelDataV2 = new MsgModelDataV2();
    ByteBuffer imgBuffer, wideImgBuffer;
    int desire;
    final WorkspaceConfiguration wsConfig = WorkspaceConfiguration.builder()
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .build();


    public ModelExecutorF2(ModelRunner modelRunner){
        this.modelRunner = modelRunner;
        ModelExecutor.instance = this;
    }

    public void ExecuteModel(Definitions.FrameData.Reader roadData, Custom.FrameBuffer.Reader roadBuf,
                             long processStartTimestamp) {

        ModelExecutor.frameWideData = frameWideData = frameData = roadData;
        ModelExecutor.msgFrameWideBuffer = msgFrameWideBuffer = msgFrameBuffer = roadBuf;

        if (stopped || initialized == false) return;

        start = System.currentTimeMillis();
        imgBuffer = updateImageBuffer(msgFrameBuffer, imgBuffer);
        wideImgBuffer = updateImageBuffer(msgFrameWideBuffer, wideImgBuffer);
        if (sh.updated("lateralPlan")){
            desire = sh.recv("lateralPlan").getLateralPlan().getDesire().ordinal();
            if (desire >= 0 && desire < CommonModelF2.DESIRE_LEN)
                desireIn[desire] = 1.0f;
        }

        for (int i=1; i<CommonModelF2.DESIRE_LEN; i++){
            if (desireIn[i] - prevDesire[i] > 0.99f)
                desireNDArr.put(0, i, desireIn[i]);
            else
                desireNDArr.put(0, i, 0.0f);
            prevDesire[i] = desireIn[i];
        }

        if (sh.updated("liveCalibration")) {
            liveCalib = sh.recv("liveCalibration").getLiveCalibration();
            PrimitiveList.Float.Reader rpy = liveCalib.getRpyCalib();
            for (int i = 0; i < 3; i++) {
                augmentRot.putScalar(i, rpy.get(i));
            }
            wrapMatrix = Preprocess.getWrapMatrix(augmentRot, Camera.cam_intrinsics, Camera.cam_intrinsics, !utils.F2, false);
            wrapMatrixWide = Preprocess.getWrapMatrix(augmentRot, Camera.cam_intrinsics, Camera.cam_intrinsics, !utils.F2, true);
        }

        netInputBuffer = imagePrepare.prepare(imgBuffer, wrapMatrix);
        netInputWideBuffer = imageWidePrepare.prepare(wideImgBuffer, wrapMatrixWide);

        if (utils.Runner == utils.USE_MODEL_RUNNER.SNPE) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsConfig, "ModelD")) {
                // NCHW to NHWC
                netInputBuffer = netInputBuffer.permute(0, 2, 3, 1).dup();
                netInputWideBuffer = netInputWideBuffer.permute(0, 2, 3, 1).dup();
            }
        }

        inputMap.put("input_imgs", netInputBuffer);
        inputMap.put("big_input_imgs", netInputWideBuffer);
        modelRunner.run(inputMap, outputMap);

        outs = parser.parser(netOutputs);

        for (int i=0; i<outs.state[0].length; i++)
            stateNDArr.put(0, i, outs.state[0][i]);

        // publish outputs
        end = System.currentTimeMillis();
        msgModelDataV2.fill(outs, processStartTimestamp, frameData.getFrameId(), -1, 0f, end - start, parser.stopSignProb);
        msgCameraOdometery.fill(outs, processStartTimestamp, frameData.getFrameId());

        ph.publishBuffer("modelV2", msgModelDataV2.serialize(frameDrops < 1));
        ph.publishBuffer("cameraOdometry", msgCameraOdometery.serialize(frameDrops < 1));

        // compute runtime stats every 10 runs
        timePerIt += end - processStartTimestamp;
        iterationNum++;
        if (iterationNum > 10) {
            ModelExecutorF3.AvgIterationTime = timePerIt / iterationNum;
            iterationNum = 0;
            timePerIt = 0;
        }

        lastFrameID = frameData.getFrameId();
    }

    INDArray netInputBuffer, netInputWideBuffer;
    INDArray wrapMatrix;
    INDArray wrapMatrixWide;
    ImagePrepare imagePrepare;
    ImagePrepare imageWidePrepare;

    public void init() {

        if (initialized)
            return;

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        if (utils.Runner == utils.USE_MODEL_RUNNER.SNPE)
            imgTensorShape = new int[]{1, 128, 256, 12}; // SNPE only supports NHWC input.

        ph.createPublishers(Arrays.asList("modelV2", "cameraOdometry"));
        sh.createSubscribers(Arrays.asList("roadCameraState", "roadCameraBuffer", "lateralPlan", "liveCalibration"));

        inputShapeMap.put("big_input_imgs", imgTensorShape);
        inputShapeMap.put("input_imgs", imgTensorShape);
        inputShapeMap.put("initial_state", stateTensorShape);
        inputShapeMap.put("desire", desireTensorShape);
        inputShapeMap.put("traffic_convention", trafficTensorShape);
        outputShapeMap.put("outputs", outputTensorShape);

        inputMap.put("initial_state", stateNDArr);
        inputMap.put("desire", desireNDArr);
        inputMap.put("traffic_convention", trafficNDArr);
        outputMap.put("outputs", netOutputs);

        modelRunner.init(inputShapeMap, outputShapeMap);
        modelRunner.warmup();

        wrapMatrix = Preprocess.getWrapMatrix(augmentRot, Camera.cam_intrinsics, Camera.cam_intrinsics, !utils.F2, false);
        wrapMatrixWide = Preprocess.getWrapMatrix(augmentRot, Camera.cam_intrinsics, Camera.cam_intrinsics, !utils.F2, true);

        // wait for a frame
        while (msgFrameBuffer == null) {
            try {
                Thread.sleep(10);
            } catch (Exception e) {}
        }

        boolean rgb;
        if (getUseGPU()){
            rgb = msgFrameBuffer.getEncoding() == Custom.FrameBuffer.Encoding.RGB;
            imagePrepare = new ImagePrepareGPU(FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1], rgb, msgFrameBuffer.getYWidth(), msgFrameBuffer.getYHeight(),
                    msgFrameBuffer.getYPixelStride(), msgFrameBuffer.getUvWidth(), msgFrameBuffer.getUvHeight(), msgFrameBuffer.getUvPixelStride(),
                    msgFrameBuffer.getUOffset(), msgFrameBuffer.getVOffset(), msgFrameBuffer.getStride());
            rgb = msgFrameWideBuffer.getEncoding() == Custom.FrameBuffer.Encoding.RGB;
            imageWidePrepare = new ImagePrepareGPU(FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1], rgb, msgFrameWideBuffer.getYWidth(), msgFrameWideBuffer.getYHeight(),
                    msgFrameWideBuffer.getYPixelStride(), msgFrameWideBuffer.getUvWidth(), msgFrameWideBuffer.getUvHeight(), msgFrameWideBuffer.getUvPixelStride(),
                    msgFrameWideBuffer.getUOffset(), msgFrameWideBuffer.getVOffset(), msgFrameWideBuffer.getStride());
        }
        else{
            rgb = msgFrameBuffer.getEncoding() == Custom.FrameBuffer.Encoding.RGB;
            imagePrepare = new ImagePrepareCPU(FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1], rgb, msgFrameBuffer.getYWidth(), msgFrameBuffer.getYHeight(),
                    msgFrameBuffer.getYPixelStride(), msgFrameBuffer.getUvWidth(), msgFrameBuffer.getUvHeight(), msgFrameBuffer.getUvPixelStride(),
                    msgFrameBuffer.getUOffset(), msgFrameBuffer.getVOffset(), msgFrameBuffer.getStride());
            rgb = msgFrameWideBuffer.getEncoding() == Custom.FrameBuffer.Encoding.RGB;
            imageWidePrepare = new ImagePrepareCPU(FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1], rgb, msgFrameWideBuffer.getYWidth(), msgFrameWideBuffer.getYHeight(),
                    msgFrameWideBuffer.getYPixelStride(), msgFrameWideBuffer.getUvWidth(), msgFrameWideBuffer.getUvHeight(), msgFrameWideBuffer.getUvPixelStride(),
                    msgFrameWideBuffer.getUOffset(), msgFrameWideBuffer.getVOffset(), msgFrameWideBuffer.getStride());
        }

        lastFrameID = frameData.getFrameId();

        initialized = true;
        params.putBool("ModelDReady", true);
    }

    public long getIterationRate() {
        return timePerIt/iterationNum;
    }

    public float getFrameDropPercent() {
        return (float)100* totalFrameDrops /(lastFrameID-firstFrameID);
    }

    public boolean isRunning() {
        return !stopped;
    }

    public boolean isInitialized(){
        return initialized;
    }

    public void dispose(){
        // dispose
        wrapMatrix.close();

        for (String inputName : inputMap.keySet()) {
            inputMap.get(inputName).close();
        }
        modelRunner.dispose();
        imagePrepare.dispose();
        ph.releaseAll();
    }

    public void stop() {
        stopped = true;
    }
    public void start(){ stopped = false; }
}
