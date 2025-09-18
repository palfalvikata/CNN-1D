# CNN1 kód

# Az egész kód egy szabad "játszótér".
# Kísérletező területek:
    # input-output kombinációk
    # rad.p görbe mérete, hol sűrűbbek a pontok
    # input-output scalerek
    # optuna tartományai
    # layerek és blokkok száma
    # aktivációs függvények
    # optimalizáló algoritmusok (adam)
    # train/test split mérete

# Könyvtárak
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # scalerek, más opciók is vannak, pl. Robust, Power, Quantile, stb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # ízlés szerint kiegészíthető egyéb metrikákkal, RMSE lentebb benne van

# NN építőelemek
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, AveragePooling1D, ReLU, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam  # optimalizálókból is van többféle, optunába be lehet tenni az ezek közti választást is
import optuna # az optuna csinálja a hiperparaméter optimalizációt. korai kereséshez 100-200 trialt ajánlok, és a kapott eredmények alapján szűkíteni a tartományait a pontosabb modell megtalálásához

# Fájlok útvonalai, írd át a sajátodra kérlek
# az input a nyomásgörbék HR alpján levágott 1 periódusa. Ezzel is lehet majd kísérletezni, hogy hány pontból álló bemenet a legjobb
# az input fájl jelenlegi alakja: az első sor a címsor, 1000 VP rad.p görbéje 1000 sorba rendezve, mind 23 pontból áll (23 oszlop)
inputs_file = "C:\\BME\\MSc\\M1\\Teamwork project\\nyomásgörbés\\maxos.xlsx"
outputs_file = "C:\\BME\\MSc\\M1\\Teamwork project\\irodalom\\VPD_gender_age.csv"
plot_dir = "C:\\BME\\PhD\\P1\\pressurek\\cnn_maxos_k_plot_age_hr_densecombined_optuna" # könyvtár definiálása a gépeden az eredményeknek
os.makedirs(plot_dir, exist_ok=True)

# beolvasás
inputs_df = pd.read_excel(inputs_file).dropna()
outputs_df = pd.read_csv(outputs_file, usecols=['k2', 'Age class', 'HR']).dropna()
min_rows = min(len(inputs_df), len(outputs_df))
inputs_df, outputs_df = inputs_df.iloc[:min_rows, :], outputs_df.iloc[:min_rows, :]
X = inputs_df.values
Y_raw = outputs_df['k2'].values.reshape(-1, 1)
Age, HR = outputs_df['Age class'].values.reshape(-1, 1), outputs_df['HR'].values.reshape(-1, 1)

# output skálázás
scaler_Y = MinMaxScaler()
Y = scaler_Y.fit_transform(Y_raw)

# train-test split. ajánlott test méretek: 0.1, 0.2, 0.3, a random state legyen egész félévben azonos, ez biztosítja, hogy mindig ugyanazokon az adatokon dolgozik
X_train, X_test, Y_train, Y_test, Age_train, Age_test, HR_train, HR_test = train_test_split(
    X, Y, Age, HR, test_size=0.2, random_state=0
)

# input skálázás
def scale_and_reshape(train, test, reshape=False):
    scaler = StandardScaler()
    train_scaled, test_scaled = scaler.fit_transform(train), scaler.transform(test)
    if reshape:  # Conv1D bemenethez
        train_scaled = train_scaled.reshape((train_scaled.shape[0], train_scaled.shape[1], 1))
        test_scaled = test_scaled.reshape((test_scaled.shape[0], test_scaled.shape[1], 1))
    return train_scaled, test_scaled

X_train_scaled, X_test_scaled = scale_and_reshape(X_train, X_test, reshape=True)
Age_train_scaled, Age_test_scaled = scale_and_reshape(Age_train, Age_test)
HR_train_scaled, HR_test_scaled = scale_and_reshape(HR_train, HR_test)

# globális lista az optuna próbálkozásokhoz
all_trials = []

# modell függvény definíció
# conv1d-batch-relu 1 blokknak számít, ezekből ki lehet próbálni többet
def build_model(filters, kernel_size, dropout_rate, dense1, dense2, learning_rate):
    signal_input = Input(shape=(X_train_scaled.shape[1], 1), name="signal_input")
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(signal_input)
    x = BatchNormalization()(x)
    x = ReLU()(x) # más aktivációs függvények is használhatóak
    x = AveragePooling1D(pool_size=2)(x)  # más pooling is lehet
    x = Flatten()(x)
    age_input, hr_input = Input(shape=(1,), name="age_input"), Input(shape=(1,), name="hr_input") # itt lehet bekötni az "extra" információt a VPD-ből a hálóba
    combined = Concatenate()([x, age_input, hr_input])
    combined = Dropout(dropout_rate)(combined)
    combined = Dense(dense1, activation='relu')(combined)
    combined = Dense(dense2, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(combined)
    model = Model(inputs=[signal_input, age_input, hr_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# optuna függvény definíció
# először nagyobb tartományokon ajánlott kísérletezni, és ahol a legjobb eredményt dobja, oda rásűríteni és úgy futtatni újra
def objective(trial):
    params = {
        "filters": trial.suggest_categorical("filters", [8, 16, 32, 64]),
        "kernel_size": trial.suggest_int("kernel_size", 2, 5),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.6),
        "dense1": trial.suggest_categorical("dense1", [16, 32, 64, 128]),
        "dense2": trial.suggest_categorical("dense2", [8, 16, 32, 64]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    }
    model = build_model(**params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    model.fit([X_train_scaled, Age_train_scaled, HR_train_scaled], Y_train,
              epochs=100, batch_size=64, verbose=0,
              validation_data=([X_test_scaled, Age_test_scaled, HR_test_scaled], Y_test),
              callbacks=[early_stopping])
    Y_pred = scaler_Y.inverse_transform(model.predict([X_test_scaled, Age_test_scaled, HR_test_scaled], verbose=0)).flatten()
    Y_test_real = scaler_Y.inverse_transform(Y_test.reshape(-1, 1)).flatten()

    # metrikák, bátran deifinálj más, neked tetszőt is akár
    r2 = r2_score(Y_test_real, Y_pred)
    mae = mean_absolute_error(Y_test_real, Y_pred)
    mse = mean_squared_error(Y_test_real, Y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((Y_test_real - Y_pred) / Y_test_real)) * 100
    all_trials.append({"trial_number": trial.number, **params, "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape})
    return rmse

# optuna futtatás
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200) # kezdeti keresésekhez 200 körül javasolt, aztán lehet kisebb

# eredmények mentése
pd.DataFrame(all_trials).to_csv(os.path.join(plot_dir, "all_trials_results.csv"), index=False)
best_trial = study.best_trial
best_params = best_trial.params
best_results = {**[t for t in all_trials if t["trial_number"] == best_trial.number][0], **best_params}
pd.DataFrame([best_results]).to_csv(os.path.join(plot_dir, "best_cnn_k2_with_age_hr_optuna_n200.csv"), index=False)

# legjobb modell tanítása újra
model = build_model(**best_params)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit([X_train_scaled, Age_train_scaled, HR_train_scaled], Y_train,
          epochs=200, batch_size=64, verbose=0,
          validation_data=([X_test_scaled, Age_test_scaled, HR_test_scaled], Y_test),
          callbacks=[early_stopping])
Y_pred = scaler_Y.inverse_transform(model.predict([X_test_scaled, Age_test_scaled, HR_test_scaled], verbose=0)).flatten()
Y_test_real = scaler_Y.inverse_transform(Y_test.reshape(-1, 1)).flatten()

# metrikák
r2, mae, mse = r2_score(Y_test_real, Y_pred), mean_absolute_error(Y_test_real, Y_pred), mean_squared_error(Y_test_real, Y_pred)
rmse, mape = np.sqrt(mse), np.mean(np.abs((Y_test_real - Y_pred) / Y_test_real)) * 100

# Bland–Altman plot, erre kérlek találj ki valami kicsit módosított saját stílust :)
def plot_bland_altman(y_true, y_pred, output_path, target_name, rmse):
    diff, mean = y_true - y_pred, (y_true + y_pred) / 2
    std, within_2std = np.std(diff), np.mean(np.abs(diff) < 1.96 * np.std(diff)) * 100
    xy, z = np.vstack([mean.ravel(), diff.ravel()]), gaussian_kde(np.vstack([mean.ravel(), diff.ravel()]))(np.vstack([mean.ravel(), diff.ravel()]))
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.scatter(mean, diff, c=z, s=20, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(im, ax=ax); cbar.set_label('Density', fontsize=12)
    for h, ls, lbl in [(0, '--', 'Mean'), (1.96*std, '-', '+1.96 SD'), (-1.96*std, '-', '-1.96 SD')]:
        ax.axhline(h, color='red', linestyle=ls, label=lbl)
    ax.text(0.05, 0.95, f'Within ±1.96 SD: {within_2std:.2f}%', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xlabel("Mean of Actual and Predicted", fontsize=16)
    ax.set_ylabel("Difference (Actual - Predicted)", fontsize=16)
    ax.set_title(f"{target_name} (RMSE={rmse:.4f})", fontsize=18)
    plt.tight_layout(); plt.savefig(output_path); plt.close()

plot_bland_altman(Y_test_real, Y_pred, os.path.join(plot_dir, "bland_altman_best_cnn_k2_optuna.png"), "k2", rmse)

print("Optuna keresés, tanítás, kiértékelés és Bland–Altman plot kész.")
