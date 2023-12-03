package ai.flow.app;

import ai.flow.app.CalibrationScreens.CalibrationInfo;
import ai.flow.common.ParamsInterface;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.ScreenAdapter;
import com.badlogic.gdx.graphics.GL20;

import static ai.flow.common.transformations.Camera.fcamIntrinsicParam;

public class SetUpScreen extends ScreenAdapter {

    FlowUI appContext;
    ParamsInterface params = ParamsInterface.getInstance();

    public SetUpScreen(FlowUI appContext) {
        this.appContext = appContext;
    }

    @Override
    public void show() {

        /*if (!params.existsAndCompare("HasAcceptedTerms", true)) {
            appContext.setScreen(new TermsScreen(appContext));
            return;
        }

        if (!params.exists("UserToken")) {
            appContext.setScreen(new RegisterScreen(appContext));
            return;
        }

        if (!params.existsAndCompare("CompletedTrainingVersion", true)){
            appContext.setScreen(new TrainingScreen(appContext));
            return;
        }*/

        if (!params.exists(fcamIntrinsicParam)){
            byte[] cameraMatrix = new byte[]{36, 120, 112, 68, 0, 0, 0, 0, -43, -117, 32, 68, 0, 0, 0, 0, 120, -73, 112, 68, 16, 87, -84, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -128, 63};
            byte[] distortionMatrix = new byte[]{10, -52, 72, 62, 36, -70, -82, -65, 121, 5, -54, -70, -51, 101, 61, 57, 112, 115, 52, 64};
            params.put("CameraMatrix", cameraMatrix);
            params.put("DistortionCoefficients", distortionMatrix);
           //appContext.launcher.startSensorD();
           //appContext.setScreen(new CalibrationInfo(appContext, false));
           //return;
        }

        appContext.launcher.startAllD();
        appContext.setScreen(new IntroScreen(appContext));
    }

    @Override
    public void render(float delta) {
        Gdx.gl.glClearColor(0, 0, 0, 1);
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
    }

    @Override
    public void hide() {
        Gdx.input.setInputProcessor(null);
        params.dispose();
    }
}
