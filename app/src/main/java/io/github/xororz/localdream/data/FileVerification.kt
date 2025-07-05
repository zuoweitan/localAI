package io.github.xororz.localdream.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.fileVerificationStore: DataStore<Preferences> by preferencesDataStore(name = "file_verification")

class FileVerification(private val context: Context) {
    private fun getFileSizeKey(modelId: String, fileName: String) =
        longPreferencesKey("${modelId}_${fileName}_size")

    suspend fun saveFileSize(modelId: String, fileName: String, size: Long) {
        context.fileVerificationStore.edit { preferences ->
            preferences[getFileSizeKey(modelId, fileName)] = size
        }
    }

    suspend fun getFileSize(modelId: String, fileName: String): Long? {
        return context.fileVerificationStore.data
            .map { preferences ->
                preferences[getFileSizeKey(modelId, fileName)]
            }
            .first()
    }

    suspend fun clearVerification(modelId: String) {
        context.fileVerificationStore.edit { preferences ->
            preferences.asMap().keys
                .filter { it.name.startsWith("${modelId}_") }
                .forEach { preferences.remove(it) }
        }
    }

    suspend fun clearFileVerification(modelId: String, fileName: String) {
        context.fileVerificationStore.edit { preferences ->
            preferences.remove(getFileSizeKey(modelId, fileName))
        }
    }
}