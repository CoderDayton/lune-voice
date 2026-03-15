use tauri::{AppHandle, Manager};

#[tauri::command]
pub async fn show_pill_window(app: AppHandle) -> Result<(), String> {
    let pill = app
        .get_webview_window("pill")
        .ok_or_else(|| "pill window not found".to_string())?;
    reposition_to_center(&app, &pill)?;
    pill.show().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn hide_pill_window(app: AppHandle) -> Result<(), String> {
    let pill = app
        .get_webview_window("pill")
        .ok_or_else(|| "pill window not found".to_string())?;
    pill.hide().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn reposition_pill_center(app: AppHandle) -> Result<(), String> {
    let pill = app
        .get_webview_window("pill")
        .ok_or_else(|| "pill window not found".to_string())?;
    reposition_to_center(&app, &pill)
}

fn reposition_to_center(_app: &AppHandle, pill: &tauri::WebviewWindow) -> Result<(), String> {
    let monitor = pill
        .current_monitor()
        .map_err(|e| e.to_string())?
        .or_else(|| pill.primary_monitor().ok().flatten())
        .ok_or_else(|| "no monitor found".to_string())?;

    let screen = monitor.size();
    let scale = monitor.scale_factor();
    let pill_size = pill.outer_size().map_err(|e| e.to_string())?;

    let screen_w = screen.width as f64 / scale;
    let screen_h = screen.height as f64 / scale;
    let pill_w = pill_size.width as f64 / scale;
    let pill_h = pill_size.height as f64 / scale;

    let x = ((screen_w - pill_w) / 2.0) as i32;
    let y = (screen_h - pill_h - 32.0) as i32; // 32px from bottom

    pill.set_position(tauri::LogicalPosition::new(x, y))
        .map_err(|e| e.to_string())?;

    Ok(())
}
