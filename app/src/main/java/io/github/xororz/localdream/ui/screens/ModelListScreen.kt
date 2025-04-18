package io.github.xororz.localdream.ui.screens

import android.content.Context
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import io.github.xororz.localdream.data.*
import io.github.xororz.localdream.navigation.Screen
import kotlinx.coroutines.launch
import java.text.DecimalFormat
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import io.github.xororz.localdream.R
import kotlinx.coroutines.withTimeout
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.PagerState
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material.icons.automirrored.filled.Help
import androidx.compose.material3.TabRowDefaults.SecondaryIndicator
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.core.content.edit

@Composable
private fun DeleteConfirmDialog(
    selectedCount: Int,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.delete_model)) },
        text = { Text(stringResource(R.string.delete_confirm, selectedCount)) },
        confirmButton = {
            TextButton(
                onClick = onConfirm,
                colors = ButtonDefaults.textButtonColors(
                    contentColor = MaterialTheme.colorScheme.error
                )
            ) {
                Text(stringResource(R.string.delete))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.cancel))
            }
        }
    )
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun ModelListScreen(
    navController: NavController,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var downloadingModel by remember { mutableStateOf<Model?>(null) }
    var currentProgress by remember { mutableStateOf<DownloadProgress?>(null) }
    var downloadError by remember { mutableStateOf<String?>(null) }
    var showDownloadConfirm by remember { mutableStateOf<Model?>(null) }
    var showDeleteConfirm by remember { mutableStateOf(false) }

    var isSelectionMode by remember { mutableStateOf(false) }
    var selectedModels by remember { mutableStateOf(setOf<Model>()) }

    val snackbarHostState = remember { SnackbarHostState() }
    val scrollBehavior =
        TopAppBarDefaults.exitUntilCollapsedScrollBehavior(rememberTopAppBarState())

    var showSettingsDialog by remember { mutableStateOf(false) }
    var tempBaseUrl by remember { mutableStateOf("") }
    val generationPreferences = remember { GenerationPreferences(context) }
    val currentBaseUrl by generationPreferences.getBaseUrl()
        .collectAsState(initial = "https://huggingface.co/")

    var version by remember { mutableStateOf(0) }
    val modelRepository = remember(version) { ModelRepository(context) }

    var showHelpDialog by remember { mutableStateOf(false) }

    val isFirstLaunch = remember {
        val preferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val isFirst = preferences.getBoolean("is_first_launch", true)
        if (isFirst) {
            preferences.edit() { putBoolean("is_first_launch", false) }
        }
        isFirst
    }

    LaunchedEffect(Unit) {
        if (isFirstLaunch) {
            showHelpDialog = true
        }
    }

    val cpuModels = remember(modelRepository.models) {
        modelRepository.models.filter { it.runOnCpu }
    }
    val npuModels = remember(modelRepository.models) {
        modelRepository.models.filter { !it.runOnCpu }
    }

    val lastViewedPage = remember {
        val preferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        preferences.getInt("last_viewed_page", 0)
    }

    val pagerState = rememberPagerState(
        initialPage = lastViewedPage,
        pageCount = { 2 }
    )

    LaunchedEffect(pagerState.currentPage) {
        val preferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        preferences.edit() { putInt("last_viewed_page", pagerState.currentPage) }
    }

    val tabTitles = listOf(
        stringResource(R.string.cpu_models),
        stringResource(R.string.npu_models)
    )

    BackHandler(enabled = isSelectionMode) {
        isSelectionMode = false
        selectedModels = emptySet()
    }
    LaunchedEffect(downloadError) {
        downloadError?.let {
            scope.launch {
                snackbarHostState.showSnackbar(
                    message = it,
                    duration = SnackbarDuration.Short
                )
                downloadError = null
            }
        }
    }
    if (showHelpDialog) {
        AlertDialog(
            onDismissRequest = { showHelpDialog = false },
            title = {
                Text(
                    text = stringResource(R.string.about_app),
                    style = MaterialTheme.typography.headlineSmall
                )
            },
            text = {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp)
                ) {
                    Text(
                        text = stringResource(R.string.must_read),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(bottom = 12.dp)
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { showHelpDialog = false }) {
                    Text(stringResource(R.string.got_it))
                }
            }
        )
    }

    if (showSettingsDialog) {
        AlertDialog(
            onDismissRequest = { showSettingsDialog = false },
            title = { Text(stringResource(R.string.settings)) },
            text = {
                Column {
                    Text(
                        stringResource(R.string.download_settings_hint),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    OutlinedTextField(
                        value = tempBaseUrl,
                        onValueChange = { tempBaseUrl = it },
                        label = { Text(stringResource(R.string.download_from)) },
                        modifier = Modifier.fillMaxWidth(),
                        placeholder = { Text("https://your-mirror-site.com/") },
                        singleLine = true
                    )

                    Spacer(modifier = Modifier.height(32.dp))
                    Text(
                        stringResource(R.string.img2img_hint),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "img2img",
                            style = MaterialTheme.typography.bodyMedium
                        )
                        val preferences = LocalContext.current.getSharedPreferences(
                            "app_prefs",
                            Context.MODE_PRIVATE
                        )
                        var useImg2img by remember {
                            mutableStateOf(preferences.getBoolean("use_img2img", true).also {
                                if (!preferences.contains("use_img2img")) {
                                    preferences.edit { putBoolean("use_img2img", true) }
                                }
                            })
                        }
                        Switch(
                            checked = useImg2img,
                            onCheckedChange = {
                                useImg2img = it
                                preferences.edit {
                                    putBoolean("use_img2img", it)
                                }
                            }
                        )
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        scope.launch {
                            if (tempBaseUrl.isNotEmpty()) {
                                generationPreferences.saveBaseUrl(tempBaseUrl)
                                modelRepository.updateBaseUrl(tempBaseUrl)
                                version += 1
                                showSettingsDialog = false
                            }
                        }
                    }
                ) {
                    Text(stringResource(R.string.confirm))
                }
            },
            dismissButton = {
                TextButton(onClick = { showSettingsDialog = false }) {
                    Text(stringResource(R.string.cancel))
                }
            }
        )
    }
    LaunchedEffect(showSettingsDialog) {
        if (showSettingsDialog) {
            tempBaseUrl = currentBaseUrl
        }
    }
    if (showDeleteConfirm && selectedModels.isNotEmpty()) {
        DeleteConfirmDialog(
            selectedCount = selectedModels.size,
            onConfirm = {
                showDeleteConfirm = false
                isSelectionMode = false

                scope.launch {
                    var successCount = 0
                    selectedModels.forEach { model ->
                        if (model.deleteModel(context)) {
                            successCount++
                        }
                    }

                    modelRepository.refreshAllModels()

                    snackbarHostState.showSnackbar(
                        if (successCount == selectedModels.size) context.getString(R.string.delete_success)
                        else context.getString(R.string.delete_failed)
                    )

                    selectedModels = emptySet()
                }
            },
            onDismiss = {
                showDeleteConfirm = false
            }
        )
    }

    showDownloadConfirm?.let { model ->
        if (downloadingModel != null) {
            AlertDialog(
                onDismissRequest = { showDownloadConfirm = null },
                title = { Text(stringResource(R.string.cannot_download)) },
                text = { Text(stringResource(R.string.cannot_download_hint)) },
                confirmButton = {
                    TextButton(onClick = { showDownloadConfirm = null }) {
                        Text(stringResource(R.string.confirm))
                    }
                }
            )
        } else {
            AlertDialog(
                onDismissRequest = { showDownloadConfirm = null },
                title = { Text(stringResource(R.string.download_model)) },
                text = {
                    if (model.isPartiallyDownloaded) {
                        Text(stringResource(R.string.continue_download_hint, model.name))
                    } else {
                        Text(stringResource(R.string.download_model_hint, model.name))
                    }
                },
                confirmButton = {
                    TextButton(
                        onClick = {
                            showDownloadConfirm = null
                            scope.launch {
                                downloadingModel = model
                                currentProgress = null

                                model.download(context).collect { result ->
                                    when (result) {
                                        is DownloadResult.Progress -> {
                                            currentProgress = result.progress
                                        }

                                        is DownloadResult.Success -> {
                                            modelRepository.refreshModelState(model.id)
                                            downloadingModel = null
                                            snackbarHostState.showSnackbar(context.getString(R.string.download_done))
                                        }

                                        is DownloadResult.Error -> {
                                            downloadingModel = null
                                            downloadError = result.message
                                        }
                                    }
                                }
                            }
                        }
                    ) {
                        Text(stringResource(R.string.confirm))
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showDownloadConfirm = null }) {
                        Text(stringResource(R.string.cancel))
                    }
                }
            )
        }
    }

    Scaffold(
        topBar = {
            LargeTopAppBar(
                title = {
                    Column {
                        Text("Local Dream✨")
                        Text(
                            if (isSelectionMode) stringResource(
                                R.string.selected_items,
                                selectedModels.size
                            ) else stringResource(R.string.available_models),
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                    }
                },
                navigationIcon = {
                    if (isSelectionMode) {
                        IconButton(onClick = {
                            isSelectionMode = false
                            selectedModels = emptySet()
                        }) {
                            Icon(Icons.Default.Close, stringResource(R.string.cancel))
                        }
                    }
                },
                actions = {
                    if (isSelectionMode && selectedModels.isNotEmpty()) {
                        IconButton(onClick = { showDeleteConfirm = true }) {
                            Icon(Icons.Default.Delete, stringResource(R.string.delete))
                        }
                    }
                    IconButton(onClick = { showHelpDialog = true }) {
                        Icon(Icons.AutoMirrored.Filled.Help, stringResource(R.string.help))
                    }
                    IconButton(onClick = { showSettingsDialog = true }) {
                        Icon(Icons.Default.Settings, stringResource(R.string.settings))
                    }
                },
                scrollBehavior = scrollBehavior
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues)
                .nestedScroll(scrollBehavior.nestedScrollConnection)
        ) {
            TabRow(
                selectedTabIndex = pagerState.currentPage,
                indicator = { tabPositions ->
                    SecondaryIndicator(
                        modifier = Modifier.tabIndicatorOffset(tabPositions[pagerState.currentPage]),
                        color = MaterialTheme.colorScheme.primary
                    )
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                tabTitles.forEachIndexed { index, title ->
                    Tab(
                        selected = pagerState.currentPage == index,
                        onClick = {
                            scope.launch {
                                pagerState.animateScrollToPage(index)
                            }
                        },
                        text = {
                            Text(
                                text = title,
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = if (pagerState.currentPage == index) FontWeight.Bold else FontWeight.Normal
                            )
                        }
                    )
                }
            }

            HorizontalPager(
                state = pagerState,
                modifier = Modifier.weight(1f)
            ) { page ->
                val models = if (page == 0) cpuModels else npuModels

                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(
                        top = 8.dp,
                        start = 16.dp,
                        end = 16.dp,
                        bottom = 16.dp
                    ),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    items(
                        items = models,
                        key = { model -> "${model.id}_${version}" }
                    ) { model ->
                        ModelCard(
                            model = model,
                            isDownloading = model == downloadingModel,
                            downloadProgress = if (model == downloadingModel) currentProgress else null,
                            isSelected = selectedModels.contains(model),
                            isSelectionMode = isSelectionMode,
                            onClick = {
                                if (!Model.isDeviceSupported() && !model.runOnCpu) {
                                    scope.launch {
                                        snackbarHostState.showSnackbar(context.getString(R.string.unsupport_npu))
                                    }
                                    return@ModelCard
                                }
                                if (isSelectionMode) {
                                    if (model.isDownloaded || model.isPartiallyDownloaded) {
                                        selectedModels = if (selectedModels.contains(model)) {
                                            selectedModels - model
                                        } else {
                                            selectedModels + model
                                        }

                                        if (selectedModels.isEmpty()) {
                                            isSelectionMode = false
                                        }
                                    }
                                } else {
                                    if (downloadingModel != null) {
                                        scope.launch {
                                            snackbarHostState.showSnackbar(context.getString(R.string.cannot_download_hint))
                                        }
                                        return@ModelCard
                                    }

                                    val isModelDownloaded = model.isDownloaded
                                    val isModelPartiallyDownloaded = model.isPartiallyDownloaded

                                    if (!isModelDownloaded) {
                                        if (isModelPartiallyDownloaded) {
                                            showDownloadConfirm = model
                                        } else {
                                            showDownloadConfirm = model
                                        }
                                    } else {
                                        navController.navigate(Screen.ModelRun.createRoute(model.id))
                                    }
                                }
                            },
                            onLongClick = {
                                if ((model.isDownloaded || model.isPartiallyDownloaded) && !isSelectionMode) {
                                    isSelectionMode = true
                                    selectedModels = setOf(model)
                                }
                            }
                        )
                    }

                    if (models.isEmpty()) {
                        item {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 32.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = if (page == 0)
                                        stringResource(R.string.no_cpu_models)
                                    else
                                        stringResource(R.string.no_npu_models),
                                    style = MaterialTheme.typography.bodyLarge,
                                    textAlign = TextAlign.Center,
                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                                )
                            }
                        }
                    }
                }
            }

            Row(
                modifier = Modifier
                    .padding(16.dp)
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.Center
            ) {
                TabPageIndicator(
                    pageCount = 2,
                    currentPage = pagerState.currentPage,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            }
        }
    }
}

@Composable
fun TabPageIndicator(
    pageCount: Int,
    currentPage: Int,
    modifier: Modifier = Modifier
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = modifier
    ) {
        repeat(pageCount) { index ->
            Box(
                modifier = Modifier
                    .size(if (currentPage == index) 10.dp else 8.dp)
                    .background(
                        color = if (currentPage == index)
                            MaterialTheme.colorScheme.primary
                        else
                            MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f),
                        shape = CircleShape
                    )
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelCard(
    model: Model,
    isDownloading: Boolean,
    downloadProgress: DownloadProgress?,
    isSelected: Boolean,
    isSelectionMode: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val elevation by animateFloatAsState(
        targetValue = if (isSelected) 8f else 1f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        )
    )

    val containerColor = when {
        isSelected -> MaterialTheme.colorScheme.secondaryContainer
        isDownloading -> MaterialTheme.colorScheme.surfaceContainerLow
        !model.isDownloaded && !model.isPartiallyDownloaded && isSelectionMode -> MaterialTheme.colorScheme.surfaceContainerLow
        else -> MaterialTheme.colorScheme.surfaceContainerLowest
    }

    val contentColor = when {
        isSelected -> MaterialTheme.colorScheme.onSecondaryContainer
        !model.isDownloaded && !model.isPartiallyDownloaded && isSelectionMode -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
        else -> MaterialTheme.colorScheme.onSurface
    }

    ElevatedCard(
        modifier = modifier
            .fillMaxWidth()
            .pointerInput(isDownloading, isSelectionMode) {
                detectTapGestures(
                    onLongPress = {
                        if ((model.isDownloaded || model.isPartiallyDownloaded) && !isDownloading && !isSelectionMode) onLongClick()
                    },
                    onTap = {
                        if (!isSelectionMode || (isSelectionMode && (model.isDownloaded || model.isPartiallyDownloaded))) {
                            onClick()
                        }
                    }
                )
            },
        colors = CardDefaults.elevatedCardColors(
            containerColor = containerColor,
            contentColor = contentColor
        ),
        elevation = CardDefaults.elevatedCardElevation(
            defaultElevation = elevation.dp
        )
    ) {
        Box(modifier = Modifier.fillMaxWidth()) {
            // CPU/NPU Badge
            Surface(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(8.dp),
                shape = RoundedCornerShape(4.dp),
                color = MaterialTheme.colorScheme.tertiaryContainer
            ) {
                Text(
                    text = if (model.runOnCpu) "CPU" else "NPU",
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onTertiaryContainer,
                    fontWeight = FontWeight.Medium
                )
            }

            // Main content
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text(
                    text = model.name,
                    style = MaterialTheme.typography.titleLarge,
                    color = contentColor
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = model.description,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    color = contentColor.copy(alpha = 0.8f)
                )
                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(R.string.model_size, model.approximateSize),
                        style = MaterialTheme.typography.labelMedium,
                        color = contentColor.copy(alpha = 0.7f)
                    )
                    when {
                        model.isDownloaded -> {
                            Text(
                                text = stringResource(R.string.downloaded),
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.Medium
                            )
                        }
                        model.isPartiallyDownloaded -> {
                            Text(
                                text = stringResource(R.string.partially_downloaded),
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.tertiary,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                }

                if (isDownloading && downloadProgress != null) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Column {
                        Text(
                            text = stringResource(
                                R.string.downloading_file,
                                downloadProgress.currentFileIndex,
                                downloadProgress.totalFiles,
                                downloadProgress.displayName
                            ),
                            style = MaterialTheme.typography.labelSmall,
                            color = contentColor.copy(alpha = 0.6f)
                        )
                        Spacer(modifier = Modifier.height(4.dp))

                        LinearProgressIndicator(
                            progress = { downloadProgress.progress },
                            modifier = Modifier.fillMaxWidth(),
                            color = MaterialTheme.colorScheme.primary,
                            trackColor = MaterialTheme.colorScheme.surfaceVariant,
                        )

                        if (downloadProgress.totalBytes > 0) {
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = "${formatFileSize(downloadProgress.downloadedBytes)} / ${
                                    formatFileSize(
                                        downloadProgress.totalBytes
                                    )
                                }",
                                style = MaterialTheme.typography.labelSmall,
                                color = contentColor.copy(alpha = 0.6f)
                            )
                        }
                    }
                }
            }
        }
    }
}

private fun formatFileSize(size: Long): String {
    val df = DecimalFormat("#.##")
    return when {
        size < 1024 -> "${size}B"
        size < 1024 * 1024 -> "${df.format(size / 1024.0)}KB"
        size < 1024 * 1024 * 1024 -> "${df.format(size / (1024.0 * 1024.0))}MB"
        else -> "${df.format(size / (1024.0 * 1024.0 * 1024.0))}GB"
    }
}