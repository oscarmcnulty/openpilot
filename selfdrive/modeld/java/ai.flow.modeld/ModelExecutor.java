package ai.flow.modeld;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ai.flow.definitions.Custom;
import ai.flow.definitions.Definitions;

public abstract class ModelExecutor {
    public final INDArray rpy_calib = Nd4j.zeros(3);
    public static Definitions.FrameData.Reader frameData;
    public static Definitions.FrameData.Reader frameWideData;
    public static Custom.FrameBuffer.Reader msgFrameBuffer;
    public static Custom.FrameBuffer.Reader msgFrameWideBuffer;
    public static ModelExecutor instance;
    public void init(){}
    public long getIterationRate(){return 0;}
    public float getFrameDropPercent() {return 0f;}
    public boolean isRunning() {return false;}
    public void ExecuteModel(Definitions.FrameData.Reader roadData, Custom.FrameBuffer.Reader roadBuf,
                             long processStartTimestamp) {}
    public boolean isInitialized() {return false;}
    public void dispose(){}
    public void stop() {}
    public void start() {}
}
