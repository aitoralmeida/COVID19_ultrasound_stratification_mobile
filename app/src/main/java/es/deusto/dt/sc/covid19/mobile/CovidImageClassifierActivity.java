package es.deusto.dt.sc.covid19.mobile;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseCustomLocalModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class CovidImageClassifierActivity extends AppCompatActivity {

    private static final String TAG = "CovidImageClassifier";

    private static final String MODEL_NAME = "covid19_model.tflite";
    private static final String LABELS_NAME = "labels.txt";
    private static final String SAMPLE_IMAGE = "59_N0_B0_P1_C1_M0_S0_F0158.png";

    private static final int IMAGE_WIDTH = 256;
    private static final int IMAGE_HEIGHT = 512;

    private FirebaseCustomLocalModel mLocalModel;
    private FirebaseModelInterpreter mInterpreter;
    private FirebaseModelInputOutputOptions mInputOutputOptions;
    private FirebaseModelInputs mInputs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initializeLocalModel();

        initializeInterpreter();

        initializeInputOutputOptions();

        // LOAD SAMPLE IMAGE TO CLASSIFY -------------------------------------------------------
        Bitmap bitmap = getBitmapFromAsset(this, SAMPLE_IMAGE);
        bitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_WIDTH, IMAGE_HEIGHT, true);

        int batchNum = 0;
        float[][][][] input = new float[1][IMAGE_WIDTH][IMAGE_HEIGHT][3];
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            for (int y = 0; y < IMAGE_HEIGHT; y++) {
                int pixel = bitmap.getPixel(x, y);
                //TODO Check if this mapping is appropriate
                input[batchNum][x][y][0] = (Color.red(pixel) - 127) / 128.0f;
                input[batchNum][x][y][1] = (Color.green(pixel) - 127) / 128.0f;
                input[batchNum][x][y][2] = (Color.blue(pixel) - 127) / 128.0f;
            }
        }
        // -------------------------------------------------------------------------------------

        addInputToModel(input);

        runInterpreter();
    }

    private void runInterpreter() {
        mInterpreter.run(mInputs, mInputOutputOptions)
                .addOnSuccessListener(
                        new OnSuccessListener<FirebaseModelOutputs>() {
                            @Override
                            public void onSuccess(FirebaseModelOutputs result) {
                                float[][] output = result.getOutput(0);
                                float[] probabilities = output[0];
                                BufferedReader reader = null;
                                try {
                                    reader = new BufferedReader(
                                            new InputStreamReader(getAssets().open(LABELS_NAME)));
                                    for (int i = 0; i < probabilities.length; i++) {
                                        String label = reader.readLine();
                                        Log.i(TAG, String.format("%s (%d): %1.4f", label, i, probabilities[i]));
                                    }
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                e.printStackTrace();
                            }
                        });
    }

    private void addInputToModel(float[][][][] input) {
        mInputs = null;
        try {
            mInputs = new FirebaseModelInputs.Builder()
                    .add(input)
                    .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
    }

    private void initializeInputOutputOptions() {
        mInputOutputOptions = null;
        try {
            mInputOutputOptions = new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, IMAGE_WIDTH, IMAGE_HEIGHT, 3})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 2})
                            .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
    }

    private void initializeInterpreter() {
        mInterpreter = null;
        try {
            FirebaseModelInterpreterOptions options =
                    new FirebaseModelInterpreterOptions.Builder(mLocalModel).build();
            mInterpreter = FirebaseModelInterpreter.getInstance(options);
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
    }

    private void initializeLocalModel() {
        mLocalModel = new FirebaseCustomLocalModel.Builder()
                .setAssetFilePath(MODEL_NAME)
                .build();
    }

    private static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return bitmap;
    }

}