package io.github.xororz.localdream.data

import android.content.Context
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.util.concurrent.TimeUnit

class DownloadManager(context: Context) {
    private val fileVerification = FileVerification(context)

    private val client = OkHttpClient.Builder()
        .retryOnConnectionFailure(true)
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    fun downloadWithResume(
        modelId: String,
        files: List<ModelFile>,
        baseUrl: String,
        modelDir: File
    ): Flow<DownloadResult> = flow {
        files.forEachIndexed { i, file ->
            val destFile = File(modelDir, file.name)
            val fileUrl = "${baseUrl.removeSuffix("/")}/${file.uri}"

            android.util.Log.d("DownloadManager", "Processing file: ${file.name}")

            var downloadedBytes = if (destFile.exists()) destFile.length() else 0L

            try {
                RandomAccessFile(destFile, "rw").use { randomAccessFile ->
                    if (downloadedBytes > 0) {
                        randomAccessFile.seek(downloadedBytes)
                    }

                    val request = Request.Builder()
                        .url(fileUrl)
                        .apply {
                            if (downloadedBytes > 0) {
                                header("Range", "bytes=$downloadedBytes-")
                            }
                        }
                        .build()

                    client.newCall(request).execute().use { response ->
                        when {
                            response.code == 416 -> {
                                val savedSize = fileVerification.getFileSize(modelId, file.name)
                                if (savedSize != null && destFile.length() == savedSize) {
                                    return@forEachIndexed
                                } else {
                                    destFile.delete()
                                    fileVerification.clearFileVerification(modelId, file.name)
                                    downloadedBytes = 0L
                                    val newRequest = Request.Builder()
                                        .url(fileUrl)
                                        .build()
                                    val newResponse = client.newCall(newRequest).execute()
                                    if (!newResponse.isSuccessful) {
                                        throw IOException("Download failed with code: ${newResponse.code}")
                                    }
                                    newResponse.use { resp ->
                                        RandomAccessFile(
                                            destFile,
                                            "rw"
                                        ).use { newRandomAccessFile ->
                                            handleResponse(
                                                resp,
                                                newRandomAccessFile,
                                                0L,
                                                file,
                                                i,
                                                files.size
                                            ) { progress ->
                                                emit(progress)
                                            }
                                        }
                                    }
                                }
                            }

                            !response.isSuccessful -> {
                                throw IOException("Download failed with code: ${response.code}")
                            }

                            else -> {
                                handleResponse(
                                    response,
                                    randomAccessFile,
                                    downloadedBytes,
                                    file,
                                    i,
                                    files.size
                                ) { progress ->
                                    emit(progress)
                                }
                            }
                        }
                    }

                    // save file size
                    val finalSize = destFile.length()
                    fileVerification.saveFileSize(modelId, file.name, finalSize)
                }
            } catch (e: Exception) {
                android.util.Log.e("DownloadManager", "Download failed for ${file.name}", e)
                destFile.delete()
                fileVerification.clearFileVerification(modelId, file.name)
                throw e
            }
        }

        emit(DownloadResult.Success)
    }

    private suspend fun handleResponse(
        response: okhttp3.Response,
        randomAccessFile: RandomAccessFile,
        initialBytes: Long,
        file: ModelFile,
        fileIndex: Int,
        totalFiles: Int,
        emitProgress: suspend (DownloadResult.Progress) -> Unit
    ) {
        var downloadedBytes = initialBytes

        response.body?.let { body ->
            val totalBytes = when {
                initialBytes > 0 -> {
                    response.header("Content-Range")
                        ?.substringAfterLast("/")
                        ?.toLongOrNull()
                        ?: (initialBytes + body.contentLength())
                }

                else -> body.contentLength()
            }

            val buffer = ByteArray(8192)
            var bytes: Int

            body.byteStream().use { inputStream ->
                while (inputStream.read(buffer).also { bytes = it } != -1) {
                    randomAccessFile.write(buffer, 0, bytes)
                    downloadedBytes += bytes

                    emitProgress(
                        DownloadResult.Progress(
                            DownloadProgress(
                                displayName = file.displayName,
                                currentFileIndex = fileIndex + 1,
                                totalFiles = totalFiles,
                                progress = if (totalBytes > 0) downloadedBytes.toFloat() / totalBytes else 0f,
                                downloadedBytes = downloadedBytes,
                                totalBytes = totalBytes
                            )
                        )
                    )
                }
            }
        }
    }
}