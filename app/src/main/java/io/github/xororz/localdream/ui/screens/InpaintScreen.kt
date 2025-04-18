package io.github.xororz.localdream.ui.screens

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.activity.compose.BackHandler
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.Base64
import kotlin.math.max

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InpaintScreen(
    originalBitmap: Bitmap,
    onInpaintComplete: (String, Bitmap, Bitmap) -> Unit,
    onCancel: () -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    val context = LocalContext.current
    val density = LocalDensity.current

    val maskBitmap = remember {
        Bitmap.createBitmap(
            originalBitmap.width,
            originalBitmap.height,
            Bitmap.Config.ARGB_8888
        ).apply {
            val canvas = Canvas(this)
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        }
    }

    val tempBitmap = remember {
        Bitmap.createBitmap(
            originalBitmap.width,
            originalBitmap.height,
            Bitmap.Config.ARGB_8888
        ).apply {
            val canvas = Canvas(this)
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        }
    }
    var displayMaskBitmap by remember { mutableStateOf(tempBitmap.asImageBitmap()) }

    val androidPath = remember { Path() }
    var brushSizeDpValue by remember { mutableStateOf(30f) }
    var isDrawing by remember { mutableStateOf(false) }
    val currentPathPoints = remember { mutableStateListOf<Offset>() }
    val pathHistory = remember { mutableStateListOf<Pair<List<Offset>, Float>>() }

    val maskPaint = remember {
        Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
            xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC)
        }
    }

    val clearPaint = remember {
        Paint().apply {
            xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
        }
    }

    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    var imageRect by remember { mutableStateOf<Rect?>(null) }
    var canvasWidth by remember { mutableStateOf(0f) }
    var canvasHeight by remember { mutableStateOf(0f) }

    fun mapToImageCoordinate(canvasPoint: Offset): Offset? {
        val rect = imageRect ?: return null
        if (!rect.contains(canvasPoint)) {
            val tolerance = 5f * density.density
            if (canvasPoint.x < rect.left - tolerance || canvasPoint.x > rect.right + tolerance ||
                canvasPoint.y < rect.top - tolerance || canvasPoint.y > rect.bottom + tolerance) {
                return null
            }
        }
        val clampedX = canvasPoint.x.coerceIn(rect.left, rect.right)
        val clampedY = canvasPoint.y.coerceIn(rect.top, rect.bottom)
        val relativeX = (clampedX - rect.left) / rect.width
        val relativeY = (clampedY - rect.top) / rect.height
        return Offset(
            relativeX * originalBitmap.width,
            relativeY * originalBitmap.height
        )
    }

    fun convertDpToImagePixels(dpValue: Float, density: Density, imageRect: Rect?, originalWidth: Int): Float {
        val rect = imageRect ?: return dpValue

        val brushSizeInScreenPx = with(density) { dpValue.dp.toPx() }

        val scale = if (rect.width > 0) originalWidth.toFloat() / rect.width else 1f

        return max(1f, brushSizeInScreenPx * scale)
    }

    fun updateDisplayMask(currentDensity: Density, currentImageRect: Rect?) {
        val canvas = Canvas(tempBitmap)
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        pathHistory.forEach { (points, pathBrushDpValue) ->
            if (points.size > 1) {
                maskPaint.strokeWidth = convertDpToImagePixels(
                    pathBrushDpValue,
                    currentDensity,
                    currentImageRect,
                    originalBitmap.width
                )
                androidPath.reset()
                androidPath.moveTo(points[0].x, points[0].y)
                for (i in 1 until points.size) {
                    androidPath.lineTo(points[i].x, points[i].y)
                }
                canvas.drawPath(androidPath, maskPaint)
            }
        }

        if (isDrawing && currentPathPoints.size > 1) {
            maskPaint.strokeWidth = convertDpToImagePixels(
                brushSizeDpValue,
                currentDensity,
                currentImageRect,
                originalBitmap.width
            )
            androidPath.reset()
            androidPath.moveTo(currentPathPoints[0].x, currentPathPoints[0].y)
            for (i in 1 until currentPathPoints.size) {
                androidPath.lineTo(currentPathPoints[i].x, currentPathPoints[i].y)
            }
            canvas.drawPath(androidPath, maskPaint)
        }

        displayMaskBitmap = tempBitmap.copy(tempBitmap.config ?: Bitmap.Config.ARGB_8888, true).asImageBitmap()
    }

    fun processMask() {
        coroutineScope.launch {
            isLoading = true
            errorMessage = null
            try {
                val finalMaskCanvas = Canvas(maskBitmap)
                finalMaskCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

                val finalPaint = Paint().apply {
                    color = Color.WHITE
                    style = Paint.Style.STROKE
                    strokeCap = Paint.Cap.ROUND
                    strokeJoin = Paint.Join.ROUND
                    isAntiAlias = true
                    xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC)
                }

                pathHistory.forEach { (points, pathBrushDpValue) ->
                    if (points.isNotEmpty()) {
                        finalPaint.strokeWidth = convertDpToImagePixels(
                            pathBrushDpValue,
                            density,
                            imageRect,
                            originalBitmap.width
                        )
                        androidPath.reset()
                        androidPath.moveTo(points[0].x, points[0].y)
                        for (i in 1 until points.size) {
                            androidPath.lineTo(points[i].x, points[i].y)
                        }
                        finalMaskCanvas.drawPath(androidPath, finalPaint)
                    }
                }

                val base64String = withContext(Dispatchers.IO) {
                    val byteArrayOutputStream = ByteArrayOutputStream()
                    maskBitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
                    val byteArray = byteArrayOutputStream.toByteArray()
                    Base64.getEncoder().encodeToString(byteArray)
                }
                onInpaintComplete(base64String, originalBitmap, maskBitmap)
            } catch (e: Exception) {
                errorMessage = "Processing mask failed: ${e.message}"
                e.printStackTrace()
            } finally {
                isLoading = false
            }
        }
    }

    fun undoLastPath(currentDensity: Density, currentImageRect: Rect?) {
        if (pathHistory.isNotEmpty()) {
            pathHistory.removeAt(pathHistory.size - 1)
            updateDisplayMask(currentDensity, currentImageRect)
        }
    }

    fun clearAllPaths(currentDensity: Density, currentImageRect: Rect?) {
        if (pathHistory.isNotEmpty()){
            pathHistory.clear()
            currentPathPoints.clear()
            isDrawing = false
            updateDisplayMask(currentDensity, currentImageRect)
        }
    }

    BackHandler { onCancel() }

    LaunchedEffect(pathHistory.size, currentPathPoints.size, isDrawing, imageRect, density) {
        updateDisplayMask(density, imageRect)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Set Inpaint Area") },
                navigationIcon = {
                    IconButton(onClick = onCancel) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                actions = {
                    IconButton(
                        onClick = {
                            if (!isLoading) processMask()
                        },
                        enabled = !isLoading
                    ) {
                        Icon(
                            imageVector = Icons.Default.Check,
                            contentDescription = "Complete Marking"
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            Column(
                modifier = Modifier.fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(16.dp)
                        .onGloballyPositioned { coordinates ->
                            canvasWidth = coordinates.size.width.toFloat()
                            canvasHeight = coordinates.size.height.toFloat()
                            val size = coordinates.size
                            val imageWidth = size.width.toFloat()
                            val imageHeight = size.height.toFloat()
                            val originalAspect = originalBitmap.width.toFloat() / originalBitmap.height.toFloat()
                            val boxAspect = imageWidth / imageHeight
                            val scaledWidth: Float
                            val scaledHeight: Float
                            if (originalAspect > boxAspect) {
                                scaledWidth = imageWidth
                                scaledHeight = imageWidth / originalAspect
                            } else {
                                scaledHeight = imageHeight
                                scaledWidth = imageHeight * originalAspect
                            }
                            val left = (imageWidth - scaledWidth) / 2
                            val top = (imageHeight - scaledHeight) / 2
                            val newRect = Rect(left, top, left + scaledWidth, bottom = top + scaledHeight)
                            if (newRect != imageRect) {
                                imageRect = newRect
                            }
                        }
                ) {
                    Image(
                        bitmap = originalBitmap.asImageBitmap(),
                        contentDescription = "Original Image",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier.fillMaxSize()
                    )

                    Canvas(
                        modifier = Modifier
                            .fillMaxSize()
                            .pointerInput(Unit) {
                                detectDragGestures(
                                    onDragStart = { offset ->
                                        val imagePoint = mapToImageCoordinate(offset)
                                        if (imagePoint != null) {
                                            isDrawing = true
                                            currentPathPoints.clear()
                                            currentPathPoints.add(imagePoint)
                                        } else {
                                            isDrawing = false
                                        }
                                    },
                                    onDrag = { change, _ ->
                                        if (isDrawing) {
                                            val imagePoint = mapToImageCoordinate(change.position)
                                            if (imagePoint != null) {
                                                currentPathPoints.add(imagePoint)
                                                change.consume()
                                            }
                                        }
                                    },
                                    onDragEnd = {
                                        if (isDrawing) {
                                            isDrawing = false
                                            if (currentPathPoints.isNotEmpty()) {
                                                pathHistory.add(Pair(currentPathPoints.toList(), brushSizeDpValue))
                                                currentPathPoints.clear()
                                            }
                                        }
                                    },
                                    onDragCancel = {
                                        if (isDrawing) {
                                            isDrawing = false
                                            if (currentPathPoints.isNotEmpty()) {
                                                pathHistory.add(Pair(currentPathPoints.toList(), brushSizeDpValue))
                                                currentPathPoints.clear()
                                            }
                                        }
                                    }
                                )
                            }
                    ) {
                        val rect = imageRect ?: return@Canvas
                        drawImage(
                            image = displayMaskBitmap,
                            srcOffset = IntOffset.Zero,
                            srcSize = IntSize(tempBitmap.width, tempBitmap.height),
                            dstOffset = IntOffset(rect.left.toInt(), rect.top.toInt()),
                            dstSize = IntSize(rect.width.toInt(), rect.height.toInt()),
                            alpha = 0.6f
                        )
                    }
                }

                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    shape = MaterialTheme.shapes.medium,
                    tonalElevation = 4.dp
                ) {
                    Column(
                        modifier = Modifier
                            .padding(16.dp)
                            .fillMaxWidth()
                    ) {
                        Text(
                            "Paint over the areas you want to inpaint",
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.padding(bottom = 12.dp)
                        )

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                "Brush Size:",
                                style = MaterialTheme.typography.labelLarge,
                                modifier = Modifier.width(100.dp)
                            )
                            Slider(
                                value = brushSizeDpValue,
                                onValueChange = { brushSizeDpValue = it },
                                valueRange = 5f..50f,
                                modifier = Modifier.weight(1f)
                            )
                            val indicatorSize = with(density) { brushSizeDpValue.dp }
                            Box(
                                modifier = Modifier
                                    .padding(start = 16.dp)
                                    .size(indicatorSize)
                                    .clip(CircleShape)
                                    .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.5f))
                            )
                        }

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 12.dp),
                            horizontalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterHorizontally)
                        ) {
                            val currentDensity = LocalDensity.current
                            val currentImageRect = imageRect

                            Button(
                                onClick = { undoLastPath(currentDensity, currentImageRect) },
                                enabled = pathHistory.isNotEmpty() && !isLoading
                            ) {
                                Icon(Icons.Default.Refresh, contentDescription = "Undo", modifier = Modifier.size(ButtonDefaults.IconSize))
                                Spacer(modifier = Modifier.width(ButtonDefaults.IconSpacing))
                                Text("Undo")
                            }

                            Button(
                                onClick = { clearAllPaths(currentDensity, currentImageRect) },
                                enabled = pathHistory.isNotEmpty() && !isLoading,
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                            ) {
                                Icon(Icons.Default.Clear, contentDescription = "Clear", modifier = Modifier.size(ButtonDefaults.IconSize))
                                Spacer(modifier = Modifier.width(ButtonDefaults.IconSpacing))
                                Text("Clear")
                            }
                        }
                    }
                }
            }

            if (isLoading) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(androidx.compose.ui.graphics.Color.Black.copy(alpha = 0.6f))
                        .pointerInput(Unit) {},
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(64.dp),
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                }
            }

            if (errorMessage != null) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = MaterialTheme.shapes.large,
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer
                        ),
                        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
                    ) {
                        Column(modifier = Modifier.padding(24.dp)) {
                            Text(
                                text = "Error",
                                style = MaterialTheme.typography.headlineSmall,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = errorMessage ?: "Unknown error occurred",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Button(
                                onClick = { errorMessage = null },
                                modifier = Modifier.align(Alignment.End),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.error,
                                    contentColor = MaterialTheme.colorScheme.onError
                                )
                            ) {
                                Text("OK")
                            }
                        }
                    }
                }
            }
        }
    }
}