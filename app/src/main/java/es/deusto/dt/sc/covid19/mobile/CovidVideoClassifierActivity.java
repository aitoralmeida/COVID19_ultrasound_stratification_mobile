package es.deusto.dt.sc.covid19.mobile;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

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

public class CovidVideoClassifierActivity extends AppCompatActivity {

    private static final String TAG = "CovidVideoClassifier";

    private static final String MODEL_NAME = "mobilenet_lstm_5_frames.tflite";
    private static final String LABELS_NAME = "labels.txt";
    private static final String SAMPLE_IMAGE = "06_N1_B0_P0_C0_M0_S0_F0044.png";

    private static final int IMAGE_WIDTH = 224;
    private static final int IMAGE_HEIGHT = 224;
    private static final int FRAMES = 5;

    private FirebaseCustomLocalModel mLocalModel;
    private FirebaseModelInterpreter mInterpreter;
    private FirebaseModelInputOutputOptions mInputOutputOptions;
    private FirebaseModelInputs mInputs;

    private Bitmap mBitmap;

    private ImageView mImageView;
    private Button mPredictButton;
    private TextView mTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = findViewById(R.id.covid_data);
        mPredictButton = findViewById(R.id.predict);
        mTextView = findViewById(R.id.prediction);

        initializeLocalModel();

        initializeInterpreter();

        initializeInputOutputOptions();

        // LOAD SAMPLE IMAGE TO CLASSIFY -------------------------------------------------------
        mBitmap = getBitmapFromAsset(this, SAMPLE_IMAGE);
        mBitmap = Bitmap.createScaledBitmap(mBitmap, IMAGE_WIDTH, IMAGE_HEIGHT, true);
        mImageView.setImageBitmap(mBitmap);
        // -------------------------------------------------------------------------------------

        mPredictButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                int num = 0;
                float[][][][][] input = new float[1][FRAMES][IMAGE_WIDTH][IMAGE_HEIGHT][3];
                for (int frame = 0; frame < FRAMES; frame++) {
                    for (int x = 0; x < IMAGE_WIDTH; x++) {
                        for (int y = 0; y < IMAGE_HEIGHT; y++) {
                            int pixel = mBitmap.getPixel(x, y);
                            //TODO Check if this mapping is appropriate
                            input[num][frame][x][y][0] = (Color.red(pixel) - 127) / 128.0f;
                            input[num][frame][x][y][1] = (Color.green(pixel) - 127) / 128.0f;
                            input[num][frame][x][y][2] = (Color.blue(pixel) - 127) / 128.0f;
                        }
                    }
                }
                addInputToModel(input);
                runInterpreter();
            }
        });
    }

    private void runInterpreter() {
        Log.d(TAG, "Running interpreter");
        mInterpreter.run(mInputs, mInputOutputOptions)
                .addOnSuccessListener(
                        new OnSuccessListener<FirebaseModelOutputs>() {
                            @Override
                            public void onSuccess(FirebaseModelOutputs result) {
                                Log.d(TAG, "Running interpreter succeeded");
                                try {
                                    float[][] output = result.getOutput(0);
                                    float[] probabilities = output[0];
                                    BufferedReader reader = new BufferedReader(new InputStreamReader(getAssets().open(LABELS_NAME)));
                                    String formattedResult = "";
                                    Log.d(TAG, "Results for " + SAMPLE_IMAGE);
                                    for (int i = 0; i < probabilities.length; i++) {
                                        String label = reader.readLine();
                                        String resultForLabel = String.format("%s (%d): %1.4f", label, i, probabilities[i]);
                                        Log.d(TAG, resultForLabel);
                                        formattedResult = formattedResult + resultForLabel + "\n";
                                    }
                                    mTextView.setText(formattedResult);
                                } catch (IOException e) {
                                    Log.d(TAG, "Could not read labels file with name " + LABELS_NAME + " from assets");
                                    e.printStackTrace();
                                }
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                Log.d(TAG, "Running interpreter FAILED");
                                e.printStackTrace();
                            }
                        });
    }

    private void addInputToModel(float[][][][][] input) {
        Log.d(TAG, "Building and adding input to model");
        try {
            mInputs = new FirebaseModelInputs.Builder()
                    .add(input)
                    .build();
        } catch (FirebaseMLException e) {
            Log.d(TAG, "Building and adding input to model FAILED");
            e.printStackTrace();
        }
    }

    private void initializeInputOutputOptions() {
        Log.d(TAG, "Building input and output options");
        try {
            mInputOutputOptions = new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, FRAMES, IMAGE_WIDTH, IMAGE_HEIGHT, 3})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 2})
                            .build();
        } catch (FirebaseMLException e) {
            Log.d(TAG, "Building input and output options FAILED");
            e.printStackTrace();
        }
    }

    private void initializeInterpreter() {
        Log.d(TAG, "Building interpreter");
        try {
            FirebaseModelInterpreterOptions options =
                    new FirebaseModelInterpreterOptions.Builder(mLocalModel).build();
            mInterpreter = FirebaseModelInterpreter.getInstance(options);
        } catch (FirebaseMLException e) {
            Log.d(TAG, "Building interpreter FAILED");
            e.printStackTrace();
        }
    }

    private void initializeLocalModel() {
        Log.d(TAG, "Getting and building local model " + MODEL_NAME + " from assets");
        mLocalModel = new FirebaseCustomLocalModel.Builder()
                .setAssetFilePath(MODEL_NAME)
                .build();
    }

    private static Bitmap getBitmapFromAsset(Context context, String filePath) {
        Log.d(TAG, "Getting bitmap of image " + filePath + " from assets");
        AssetManager assetManager = context.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            Log.d(TAG, "Could not read image file with name " + filePath + " from assets");
            e.printStackTrace();
        }

        if (bitmap != null) {
            Log.d(TAG, "Returning bitmap of image " + filePath + " from assets");
        }
        else {
            Log.d(TAG, "Could not retrieve bitmap of image " + filePath + " from assets... Returning null");
        }

        return bitmap;
    }

}