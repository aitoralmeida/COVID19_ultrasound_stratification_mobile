package es.deusto.dt.sc.covid19.mobile;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class CovidVideoClassifierActivity extends AppCompatActivity {

    private static final String TAG = "CovidImageClassifier";

    private static final String MODEL_NAME = "mobilenet_lstm_covid";
    private static final String LABELS_NAME = "labels.txt";
    private static final String SAMPLE_IMAGE = "06_N1_B0_P0_C0_M0_S0_F0044.png";

    private static final int IMAGE_WIDTH = 224;
    private static final int IMAGE_HEIGHT = 224;
    private static final int CHANNELS = 3;
    private static final int FRAMES = 5;

    private static final int CLASSES = 2;

    private static final int NUM_THREADS = 4;

    private FirebaseCustomRemoteModel mRemoteModel;
    private FirebaseModelDownloadConditions mModelDownloadConditions;
    private Interpreter mInterpreter;

    private Bitmap mBitmap;

    private ByteBuffer mInput;
    private ByteBuffer mOutput;

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

        mRemoteModel = new FirebaseCustomRemoteModel.Builder(MODEL_NAME).build();
        mModelDownloadConditions = new FirebaseModelDownloadConditions.Builder().requireWifi().build();
        FirebaseModelManager.getInstance().download(mRemoteModel, mModelDownloadConditions)
                .addOnSuccessListener(new OnSuccessListener<Void>() {
                    @Override
                    public void onSuccess(Void v) {
                        // Download complete. Depending on your app, you could enable
                        // the ML feature, or switch from the local model to the remote
                        // model, etc.
                        Log.d(TAG, "Downloaded model!");
                    }
                });
        FirebaseModelManager.getInstance().getLatestModelFile(mRemoteModel)
                .addOnCompleteListener(new OnCompleteListener<File>() {
                    @Override
                    public void onComplete(@NonNull Task<File> task) {
                        File modelFile = task.getResult();
                        if (modelFile != null) {
                            Interpreter.Options options = new Interpreter.Options();
                            options.setNumThreads(NUM_THREADS);
                            mInterpreter = new Interpreter(modelFile, options);
                            Log.d(TAG, "Constructed interpreter!");
                        }
                    }
                });

        // LOAD SAMPLE IMAGE TO CLASSIFY -------------------------------------------------------
        mBitmap = getBitmapFromAsset(this, SAMPLE_IMAGE);
        mBitmap = Bitmap.createScaledBitmap(mBitmap, IMAGE_WIDTH, IMAGE_HEIGHT, true);
        mImageView.setImageBitmap(mBitmap);
        // -------------------------------------------------------------------------------------

        mInput = ByteBuffer.allocateDirect(IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS * FRAMES * 4).order(ByteOrder.nativeOrder());
        for (int frame = 0; frame < FRAMES; frame++) {
            for (int y = 0; y < IMAGE_WIDTH; y++) {
                for (int x = 0; x < IMAGE_HEIGHT; x++) {
                    int px = mBitmap.getPixel(x, y);
                    // Get channel values from the pixel value.
                    int r = Color.red(px);
                    int g = Color.green(px);
                    int b = Color.blue(px);
                    // Normalize channel values to [-1.0, 1.0]. This requirement depends
                    // on the model. For example, some models might require values to be
                    // normalized to the range [0.0, 1.0] instead.
                    float rf = (r - 127) / 255.0f;
                    float gf = (g - 127) / 255.0f;
                    float bf = (b - 127) / 255.0f;
                    mInput.putFloat(rf);
                    mInput.putFloat(gf);
                    mInput.putFloat(bf);
                }
            }
        }

        int bufferSize = CLASSES * Float.SIZE / Byte.SIZE;
        mOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());

        mPredictButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                runInterpreter();
            }
        });

    }

    private void runInterpreter() {
        Log.d(TAG, "Running interpreter");
        final long startTime = SystemClock.uptimeMillis();
        mInterpreter.run(mInput, mOutput);
        final long inferenceTime = SystemClock.uptimeMillis() - startTime;
        Log.d(TAG, "Running interpreter succeeded");
        Log.d(TAG, "Inference time: " + inferenceTime);
        mOutput.rewind();
        FloatBuffer probabilities = mOutput.asFloatBuffer();
        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(getAssets().open(LABELS_NAME)));
            String formattedResult = "";
            Log.d(TAG, "Results for " + SAMPLE_IMAGE);
            for (int i = 0; i < probabilities.capacity(); i++) {
                String label = reader.readLine();
                String resultForLabel = String.format("%s (%d): %1.4f", label, i, probabilities.get(i));
                Log.d(TAG, resultForLabel);
                formattedResult = formattedResult + resultForLabel + "\n";
            }
            mTextView.setText(formattedResult);
        } catch (IOException e) {
            // File not found?
        }
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