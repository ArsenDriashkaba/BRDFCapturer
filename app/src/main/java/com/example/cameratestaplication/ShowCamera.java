package com.example.cameratestaplication;

import android.content.Context;
import android.content.res.Configuration;
import android.hardware.Camera;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import androidx.annotation.NonNull;

import java.io.IOException;
import java.util.List;

public class ShowCamera extends SurfaceView implements SurfaceHolder.Callback {

    Camera camera;
    SurfaceHolder holder;

    public ShowCamera (Context context, Camera camera){
        super(context);
        this.camera = camera;
        holder = getHolder();
        holder.addCallback(this);

    }

    @Override
    public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {

    }

    @Override
    public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
        camera.stopPreview();
        camera.release();
    }

    @Override
    public void surfaceCreated(@NonNull SurfaceHolder holder) {
        Camera.Parameters params = camera.getParameters();
        // skusime zapnut HDR mod
        params.setSceneMode(Camera.Parameters.SCENE_MODE_HDR);

        // Najdeme podporovane rozlisenia camery nasho mobilu
        List<Camera.Size> resolutionSizes = params.getSupportedPictureSizes();
        Camera.Size pictureSize = null;
        for (Camera.Size size : resolutionSizes){
            pictureSize = size;
        }

        // Zmenime orientaciu camery
        if (this.getResources().getConfiguration().orientation != Configuration.ORIENTATION_LANDSCAPE){
            params.set("orientation", "portrait");
            camera.setDisplayOrientation(90);
            params.setRotation(90);
        }
        else{
            params.set("orientation", "landscape");
            camera.setDisplayOrientation(0);
            params.setRotation(0);
        }

        // Nastavime vysku a dlzku rozlisenia nasej camery
        params.setPictureSize(pictureSize.width, pictureSize.height);

        camera.setParameters(params);
        try {
            camera.setPreviewDisplay(holder);
            camera.startPreview();
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }
}
