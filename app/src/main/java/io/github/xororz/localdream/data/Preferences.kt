package io.github.xororz.localdream.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import java.io.IOException

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "generation_prefs")

class GenerationPreferences(private val context: Context) {
    private fun getPromptKey(modelId: String) = stringPreferencesKey("${modelId}_prompt")
    private fun getNegativePromptKey(modelId: String) =
        stringPreferencesKey("${modelId}_negative_prompt")

    private fun getStepsKey(modelId: String) = floatPreferencesKey("${modelId}_steps")
    private fun getCfgKey(modelId: String) = floatPreferencesKey("${modelId}_cfg")
    private fun getSeedKey(modelId: String) = stringPreferencesKey("${modelId}_seed")
    private fun getSizeKey(modelId: String) = intPreferencesKey("${modelId}_size")
    private val BASE_URL_KEY = stringPreferencesKey("base_url")

    suspend fun saveBaseUrl(url: String) {
        context.dataStore.edit { preferences ->
            preferences[BASE_URL_KEY] = url
        }
    }

    fun getBaseUrl(): Flow<String> {
        return context.dataStore.data
            .map { preferences ->
                preferences[BASE_URL_KEY] ?: "https://huggingface.co/"
            }
    }

    suspend fun savePrompt(modelId: String, prompt: String) {
        context.dataStore.edit { preferences ->
            preferences[getPromptKey(modelId)] = prompt
        }
    }

    suspend fun saveNegativePrompt(modelId: String, negativePrompt: String) {
        context.dataStore.edit { preferences ->
            preferences[getNegativePromptKey(modelId)] = negativePrompt
        }
    }

    suspend fun saveSteps(modelId: String, steps: Float) {
        context.dataStore.edit { preferences ->
            preferences[getStepsKey(modelId)] = steps
        }
    }

    suspend fun saveCfg(modelId: String, cfg: Float) {
        context.dataStore.edit { preferences ->
            preferences[getCfgKey(modelId)] = cfg
        }
    }

    suspend fun saveSeed(modelId: String, seed: String) {
        context.dataStore.edit { preferences ->
            preferences[getSeedKey(modelId)] = seed
        }
    }

    suspend fun saveSize(modelId: String, size: Int) {
        context.dataStore.edit { preferences ->
            preferences[getSizeKey(modelId)] = size
        }
    }

    fun getPreferences(modelId: String): Flow<GenerationPrefs> {
        return context.dataStore.data
            .catch { exception ->
                if (exception is IOException) {
                    emit(emptyPreferences())
                } else {
                    throw exception
                }
            }
            .map { preferences ->
                GenerationPrefs(
                    prompt = preferences[getPromptKey(modelId)] ?: "",
                    negativePrompt = preferences[getNegativePromptKey(modelId)] ?: "",
                    steps = preferences[getStepsKey(modelId)] ?: 20f,
                    cfg = preferences[getCfgKey(modelId)] ?: 7f,
                    seed = preferences[getSeedKey(modelId)] ?: "",
                    size = preferences[getSizeKey(modelId)] ?: 256
                )
            }
    }
}

data class GenerationPrefs(
    val prompt: String = "",
    val negativePrompt: String = "",
    val steps: Float = 20f,
    val cfg: Float = 7f,
    val seed: String = "",
    val size: Int = 512
)