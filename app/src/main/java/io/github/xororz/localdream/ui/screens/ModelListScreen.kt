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
import androidx.compose.foundation.BorderStroke
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
import androidx.compose.ui.draw.clip
import androidx.core.content.edit
import java.io.File

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

    var showHighresDownloadConfirm by remember { mutableStateOf<Pair<Model, Int>?>(null) }
    var downloadingHighres by remember { mutableStateOf<Pair<String, Int>?>(null) }
    var highresProgress by remember { mutableStateOf<DownloadProgress?>(null) }
    var showHighres404Dialog by remember { mutableStateOf<String?>(null) }

    var isSelectionMode by remember { mutableStateOf(false) }
    var selectedModels by remember { mutableStateOf(setOf<Model>()) }

    val snackbarHostState = remember { SnackbarHostState() }
    val scrollBehavior =
        TopAppBarDefaults.exitUntilCollapsedScrollBehavior(rememberTopAppBarState())

    var showSettingsDialog by remember { mutableStateOf(false) }
    var showFileManagerDialog by remember { mutableStateOf(false) }
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
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(24.dp)
                ) {
                    // Download source settings section
                    Column {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.padding(bottom = 12.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.CloudDownload,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(20.dp)
                            )
                            Text(
                                stringResource(R.string.download_source),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Medium
                            )
                        }
                        Text(
                            stringResource(R.string.download_settings_hint),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                            modifier = Modifier.padding(bottom = 12.dp)
                        )
                        OutlinedTextField(
                            value = tempBaseUrl,
                            onValueChange = { tempBaseUrl = it },
                            label = { Text(stringResource(R.string.download_from)) },
                            modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text("https://your-mirror-site.com/") },
                            singleLine = true
                        )
                    }

                    // Feature settings section
                    Column {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.padding(bottom = 12.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Tune,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(20.dp)
                            )
                            Text(
                                stringResource(R.string.feature_settings),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Medium
                            )
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Column(
                                modifier = Modifier.weight(1f)
                            ) {
                                Text(
                                    text = "img2img",
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.Medium
                                )
                                Text(
                                    stringResource(R.string.img2img_hint),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                                )
                            }
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

                    // File management section
                    Column {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.padding(bottom = 12.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Folder,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(20.dp)
                            )
                            Text(
                                stringResource(R.string.file_management),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Medium
                            )
                        }
                        OutlinedButton(
                            onClick = {
                                showSettingsDialog = false
                                showFileManagerDialog = true
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(
                                imageVector = Icons.Default.FolderOpen,
                                contentDescription = null,
                                modifier = Modifier.padding(end = 8.dp)
                            )
                            Text(stringResource(R.string.file_manager))
                        }
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

    if (showFileManagerDialog) {
        FileManagerDialog(
            context = context,
            onDismiss = { showFileManagerDialog = false },
            onFileDeleted = {
                modelRepository.refreshAllModels()
                scope.launch {
                    snackbarHostState.showSnackbar(context.getString(R.string.file_deleted))
                }
            }
        )
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
    showHighres404Dialog?.let { errorMessage ->
        AlertDialog(
            onDismissRequest = { showHighres404Dialog = null },
            title = {
                Text(
                    text = "High Resolution Patch Not Found",
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.error
                )
            },
            text = {
                Text(
                    text = errorMessage,
                    style = MaterialTheme.typography.bodyMedium
                )
            },
            confirmButton = {
                TextButton(onClick = { showHighres404Dialog = null }) {
                    Text("Got it")
                }
            },
            icon = {
                Icon(
                    imageVector = Icons.Default.Error,
                    contentDescription = "Error",
                    tint = MaterialTheme.colorScheme.error
                )
            }
        )
    }
    showHighresDownloadConfirm?.let { (model, resolution) ->
        AlertDialog(
            onDismissRequest = { showHighresDownloadConfirm = null },
            title = {
                Text(stringResource(R.string.download_patch))
            },
            text = {
                Text(
                    stringResource(
                        R.string.download_patch_hint,
                        resolution,
                        model.name,
                    )
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showHighresDownloadConfirm = null
                        scope.launch {
                            downloadingHighres = Pair(model.id, resolution)
                            highresProgress = null

                            model.downloadHighresPatch(context, resolution).collect { result ->
                                when (result) {
                                    is DownloadResult.Progress -> {
                                        highresProgress = result.progress
                                    }

                                    is DownloadResult.Success -> {
                                        modelRepository.refreshModelState(model.id)
                                        downloadingHighres = null
                                        highresProgress = null
                                        snackbarHostState.showSnackbar("${resolution}px patch downloaded successfully")
                                    }

                                    is DownloadResult.Error -> {
                                        downloadingHighres = null
                                        highresProgress = null

                                        if (result.message.startsWith("PATCH_NOT_FOUND|")) {
                                            showHighres404Dialog =
                                                result.message.substringAfter("PATCH_NOT_FOUND|")
                                        } else {
                                            downloadError = result.message
                                        }
                                    }
                                }
                            }
                        }
                    }
                ) {
                    Text("Confirm")
                }
            },
            dismissButton = {
                TextButton(onClick = { showHighresDownloadConfirm = null }) {
                    Text("Cancel")
                }
            }
        )
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
                            downloadingHighres = downloadingHighres?.let { (modelId, resolution) ->
                                if (modelId == model.id) resolution else null
                            },
                            highresProgress = if (downloadingHighres?.first == model.id) highresProgress else null,
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
                                    if (downloadingModel != null || downloadingHighres != null) {
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
                            },
                            onHighresDownload = { resolution ->
                                if (downloadingModel == null && downloadingHighres == null) {
                                    showHighresDownloadConfirm = Pair(model, resolution)
                                }
                            },
                            onHighresClick = { resolution ->
                                if (downloadingModel != null || downloadingHighres != null) {
                                    scope.launch {
                                        snackbarHostState.showSnackbar(context.getString(R.string.cannot_download_hint))
                                    }
                                    return@ModelCard
                                }
                                navController.navigate(Screen.ModelRun.createRoute("${model.id}?resolution=$resolution"))
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
    onHighresDownload: (Int) -> Unit,
    onHighresClick: (Int) -> Unit,
    downloadingHighres: Int? = null,
    highresProgress: DownloadProgress? = null,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val elevation by animateFloatAsState(
        targetValue = if (isSelected) 8f else 1f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "CardElevationAnimation"
    )

    val containerColor = when {
        isSelected -> MaterialTheme.colorScheme.secondaryContainer
        isDownloading -> MaterialTheme.colorScheme.surfaceContainerLow
        !model.isDownloaded && !model.isPartiallyDownloaded && isSelectionMode -> MaterialTheme.colorScheme.surfaceContainerLow
        else -> MaterialTheme.colorScheme.surfaceContainerLowest
    }

    val contentColor = when {
        isSelected -> MaterialTheme.colorScheme.onSecondaryContainer
        !model.isDownloaded && !model.isPartiallyDownloaded && isSelectionMode -> MaterialTheme.colorScheme.onSurface.copy(
            alpha = 0.5f
        )

        else -> MaterialTheme.colorScheme.onSurface
    }

    val backgroundColor by animateColorAsState(
        targetValue = containerColor,
        animationSpec = tween(durationMillis = 300),
        label = "CardBackgroundColorAnimation"
    )

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
            containerColor = backgroundColor,
            contentColor = contentColor
        ),
        elevation = CardDefaults.elevatedCardElevation(
            defaultElevation = elevation.dp
        ),
        shape = RoundedCornerShape(16.dp)
    ) {
        Box(modifier = Modifier.fillMaxWidth()) {
            Surface(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(8.dp),
                shape = RoundedCornerShape(4.dp),
                color = if (model.runOnCpu)
                    MaterialTheme.colorScheme.tertiaryContainer
                else
                    MaterialTheme.colorScheme.primaryContainer
            ) {
                Text(
                    text = if (model.runOnCpu) "CPU" else "NPU",
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = if (model.runOnCpu)
                        MaterialTheme.colorScheme.onTertiaryContainer
                    else
                        MaterialTheme.colorScheme.onPrimaryContainer,
                    fontWeight = FontWeight.Medium
                )
            }

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
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.weight(1f)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.SdStorage,
                                contentDescription = "model size",
                                tint = contentColor.copy(alpha = 0.6f),
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                text = model.approximateSize,
                                style = MaterialTheme.typography.labelMedium,
                                color = contentColor.copy(alpha = 0.7f)
                            )
                        }

                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.AspectRatio,
                                contentDescription = "image size",
                                tint = contentColor.copy(alpha = 0.6f),
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                text = if (model.runOnCpu) "128~512" else "${model.generationSize}×${model.generationSize}",
                                style = MaterialTheme.typography.labelMedium,
                                color = contentColor.copy(alpha = 0.7f)
                            )
                        }
                    }

                    when {
                        model.isDownloaded -> {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.CheckCircle,
                                    contentDescription = "downloaded",
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(16.dp)
                                )
                                Text(
                                    text = stringResource(R.string.downloaded),
                                    style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.primary,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }

                        model.isPartiallyDownloaded -> {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Sync,
                                    contentDescription = "partially downloaded",
                                    tint = MaterialTheme.colorScheme.tertiary,
                                    modifier = Modifier.size(16.dp)
                                )
                                Text(
                                    text = stringResource(R.string.partially_downloaded),
                                    style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.tertiary,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }

                        else -> {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.CloudDownload,
                                    contentDescription = "download",
                                    tint = contentColor.copy(alpha = 0.6f),
                                    modifier = Modifier.size(16.dp)
                                )
                                Text(
                                    text = stringResource(R.string.download),
                                    style = MaterialTheme.typography.labelMedium,
                                    color = contentColor.copy(alpha = 0.6f)
                                )
                            }
                        }
                    }
                }

                if (!model.runOnCpu && model.isDownloaded && model.supportedHighres.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(12.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = stringResource(R.string.highres_patch),
                            style = MaterialTheme.typography.labelMedium,
                            color = contentColor.copy(alpha = 0.8f),
                            fontWeight = FontWeight.Medium
                        )

                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            model.supportedHighres.forEach { resolution ->
                                HighresButton(
                                    resolution = resolution,
                                    isDownloaded = model.isHighresDownloaded(context, resolution),
                                    isDownloading = downloadingHighres == resolution,
                                    isEnabled = !isDownloading && downloadingHighres == null,
                                    onClick = {
                                        if (model.isHighresDownloaded(context, resolution)) {
                                            onHighresClick(resolution)
                                        } else {
                                            onHighresDownload(resolution)
                                        }
                                    }
                                )
                            }
                        }
                    }
                }

                if (isDownloading && downloadProgress != null) {
                    Spacer(modifier = Modifier.height(8.dp))
                    DownloadProgressIndicator(
                        progress = downloadProgress,
                        contentColor = contentColor,
                        model = model
                    )
                }

                if (downloadingHighres != null && highresProgress != null) {
                    Spacer(modifier = Modifier.height(8.dp))
                    DownloadProgressIndicator(
                        progress = highresProgress,
                        contentColor = contentColor,
                        model = model,
                        isHighres = true
                    )
                }
            }
        }
    }
}

@Composable
private fun HighresButton(
    resolution: Int,
    isDownloaded: Boolean,
    isDownloading: Boolean,
    isEnabled: Boolean,
    onClick: () -> Unit
) {
    val buttonColors = when {
        isDownloaded -> ButtonDefaults.filledTonalButtonColors(
            containerColor = MaterialTheme.colorScheme.primary,
            contentColor = MaterialTheme.colorScheme.onPrimary
        )

        isDownloading -> ButtonDefaults.filledTonalButtonColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
            contentColor = MaterialTheme.colorScheme.onSurfaceVariant
        )

        else -> ButtonDefaults.outlinedButtonColors(
            contentColor = MaterialTheme.colorScheme.onSurface.copy(alpha = if (isEnabled) 1f else 0.5f)
        )
    }

    if (isDownloaded) {
        FilledTonalButton(
            onClick = onClick,
            enabled = isEnabled,
            colors = buttonColors,
            modifier = Modifier.height(28.dp),
            contentPadding = PaddingValues(horizontal = 8.dp)
        ) {
            Text(
                text = "${resolution}px",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Medium
            )
        }
    } else {
        OutlinedButton(
            onClick = onClick,
            enabled = isEnabled && !isDownloading,
            colors = buttonColors,
            modifier = Modifier.height(28.dp),
            contentPadding = PaddingValues(horizontal = 8.dp),
            border = BorderStroke(
                width = 1.dp,
                color = if (isEnabled && !isDownloading)
                    MaterialTheme.colorScheme.outline
                else
                    MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
            )
        ) {
            if (isDownloading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(10.dp),
                    strokeWidth = 1.dp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.width(3.dp))
            }
            Text(
                text = "${resolution}px",
                style = MaterialTheme.typography.labelSmall
            )
        }
    }
}

@Composable
private fun DownloadProgressIndicator(
    progress: DownloadProgress,
    contentColor: Color,
    model: Model,
    isHighres: Boolean = false
) {
    Column {
        Text(
            text = if (isHighres) {
                stringResource(
                    R.string.downloading_file,
                    1,
                    1,
                    progress.displayName
                )
            } else {
                stringResource(
                    R.string.downloading_file,
                    progress.currentFileIndex,
                    progress.totalFiles,
                    progress.displayName
                )
            },
            style = MaterialTheme.typography.labelSmall,
            color = contentColor.copy(alpha = 0.6f)
        )
        Spacer(modifier = Modifier.height(4.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(4.dp)
                .clip(RoundedCornerShape(3.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(progress.progress)
                    .fillMaxHeight()
                    .background(
                        if (model.runOnCpu)
                            MaterialTheme.colorScheme.tertiary
                        else
                            MaterialTheme.colorScheme.primary
                    )
            )
        }

        if (progress.totalBytes > 0) {
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = "${formatFileSize(progress.downloadedBytes)} / ${
                    formatFileSize(progress.totalBytes)
                }",
                style = MaterialTheme.typography.labelSmall,
                color = contentColor.copy(alpha = 0.6f)
            )
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

@Composable
private fun FileManagerDialog(
    context: Context,
    onDismiss: () -> Unit,
    onFileDeleted: () -> Unit
) {
    var modelFolders by remember { mutableStateOf<List<Pair<String, Int>>>(emptyList()) }
    var selectedFolder by remember { mutableStateOf<String?>(null) }
    var folderFiles by remember { mutableStateOf<List<File>>(emptyList()) }
    var showDeleteConfirm by remember { mutableStateOf<File?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    val fileVerification = remember { FileVerification(context) }
    val scope = rememberCoroutineScope()

    fun loadFolders() {
        val modelsDir = Model.getModelsDir(context)
        val folders = mutableListOf<Pair<String, Int>>()

        if (modelsDir.exists() && modelsDir.isDirectory) {
            modelsDir.listFiles()?.forEach { modelDir ->
                if (modelDir.isDirectory) {
                    val fileCount = modelDir.listFiles()?.size ?: 0
                    if (fileCount > 0) {
                        folders.add(Pair(modelDir.name, fileCount))
                    }
                }
            }
        }
        modelFolders = folders
        isLoading = false
    }

    fun loadFilesForFolder(folderName: String) {
        val modelsDir = Model.getModelsDir(context)
        val folderDir = File(modelsDir, folderName)
        folderFiles = folderDir.listFiles()?.toList() ?: emptyList()
    }

    LaunchedEffect(Unit) {
        loadFolders()
    }

    if (showDeleteConfirm != null) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = null },
            title = { Text(stringResource(R.string.delete_file)) },
            text = { Text(stringResource(R.string.delete_file_confirm, showDeleteConfirm!!.name)) },
            confirmButton = {
                TextButton(
                    onClick = {
                        val fileToDelete = showDeleteConfirm!!
                        if (fileToDelete.delete()) {
                            selectedFolder?.let { modelId ->
                                scope.launch {
                                    fileVerification.clearFileVerification(
                                        modelId,
                                        fileToDelete.name
                                    )
                                }
                            }
                            onFileDeleted()
                            // Reload current folder files
                            selectedFolder?.let { loadFilesForFolder(it) }
                            // Reload folder list to update file count
                            loadFolders()
                        }
                        showDeleteConfirm = null
                    },
                    colors = ButtonDefaults.textButtonColors(
                        contentColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Text(stringResource(R.string.delete))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = null }) {
                    Text(stringResource(R.string.cancel))
                }
            }
        )
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                if (selectedFolder != null) {
                    IconButton(
                        onClick = { selectedFolder = null },
                        modifier = Modifier.size(24.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = stringResource(R.string.back_to_folders),
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
                Text(
                    text = selectedFolder?.let {
                        stringResource(R.string.model_folder, it)
                    } ?: stringResource(R.string.file_manager),
                    style = MaterialTheme.typography.headlineSmall
                )
            }
        },
        text = {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(400.dp)
            ) {
                if (isLoading) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator()
                        Text(
                            stringResource(R.string.loading_files),
                            modifier = Modifier.padding(top = 48.dp),
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                } else if (selectedFolder == null) {
                    // Show folder list
                    if (modelFolders.isEmpty()) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Icon(
                                    imageVector = Icons.Default.FolderOpen,
                                    contentDescription = null,
                                    modifier = Modifier.size(48.dp),
                                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
                                )
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    stringResource(R.string.no_model_files),
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                                )
                            }
                        }
                    } else {
                        LazyColumn(
                            modifier = Modifier.fillMaxSize(),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            items(modelFolders) { (folderName, fileCount) ->
                                Card(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .pointerInput(Unit) {
                                            detectTapGestures(
                                                onTap = {
                                                    selectedFolder = folderName
                                                    loadFilesForFolder(folderName)
                                                }
                                            )
                                        },
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                                    )
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(16.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.Folder,
                                                contentDescription = null,
                                                tint = MaterialTheme.colorScheme.primary
                                            )
                                            Column {
                                                Text(
                                                    text = folderName,
                                                    style = MaterialTheme.typography.bodyLarge,
                                                    fontWeight = FontWeight.Medium
                                                )
                                                Text(
                                                    text = stringResource(
                                                        R.string.file_count,
                                                        fileCount
                                                    ),
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurface.copy(
                                                        alpha = 0.6f
                                                    )
                                                )
                                            }
                                        }
                                        Icon(
                                            imageVector = Icons.Default.ChevronRight,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                                        )
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Show files in selected folder
                    if (folderFiles.isEmpty()) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                stringResource(R.string.no_model_files),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                            )
                        }
                    } else {
                        LazyColumn(
                            modifier = Modifier.fillMaxSize(),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            items(folderFiles) { file ->
                                Card(
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                                    )
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(12.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                                            modifier = Modifier.weight(1f)
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.InsertDriveFile,
                                                contentDescription = null,
                                                tint = MaterialTheme.colorScheme.secondary
                                            )
                                            Column {
                                                Text(
                                                    text = file.name,
                                                    style = MaterialTheme.typography.bodyMedium,
                                                    fontWeight = FontWeight.Medium
                                                )
                                                Text(
                                                    text = formatFileSize(file.length()),
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurface.copy(
                                                        alpha = 0.6f
                                                    )
                                                )
                                            }
                                        }

                                        IconButton(
                                            onClick = { showDeleteConfirm = file },
                                            colors = IconButtonDefaults.iconButtonColors(
                                                contentColor = MaterialTheme.colorScheme.error
                                            )
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.Delete,
                                                contentDescription = stringResource(R.string.delete_file)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.close))
            }
        }
    )
}