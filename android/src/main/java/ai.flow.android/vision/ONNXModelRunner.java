package ai.flow.android.vision;

import ai.flow.modeld.ModelRunner;
import ai.onnxruntime.*;
import ai.onnxruntime.providers.NNAPIFlags;
import onnx.Onnx;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

public class ONNXModelRunner extends ModelRunner {
    String modelPath;
    Map<String, OnnxTensor> container = new HashMap<>();
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    OrtSession session;
    Map<String, long[]> shapes = new HashMap<>();

    boolean useGPU;

    public ONNXModelRunner(String modelPath, boolean useGPU){
        this.modelPath = modelPath;
        this.useGPU = useGPU;
    }

    @Override
    public void init(Map<String, int[]> shapes, Map<String, int[]> outputShapes) {
        try {
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            opts.addNnapi();
            session = env.createSession(modelPath + ".ort", opts);
            } catch (OrtException e) {
                throw new RuntimeException(e);
        }

        for (String name : shapes.keySet()) {
            this.shapes.put(name, Arrays.stream(shapes.get(name)).mapToLong((i) -> (long) i).toArray());
        }
    }

    @Override
    public void run(Map<String, INDArray> inputMap, Map<String, float[]> outputMap) {
        float[] netOutputs = null;
        try {
            for (String inputName : inputMap.keySet()) {
                container.put(inputName, OnnxTensor.createTensor(env, inputMap.get(inputName).data().asNioFloat(), shapes.get(inputName)));
            }
            try (OrtSession.Result netOutputsTensor = session.run(container);){
                netOutputs = ((float[][])netOutputsTensor.get(0).getValue())[0];
            } catch(OrtException e){
                System.out.println(e);
            }

        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        assert netOutputs != null;
        System.arraycopy(netOutputs, 0, outputMap.get("outputs"), 0, outputMap.get("outputs").length);
    }

    @Override
    public void dispose(){
        try {
            env.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
