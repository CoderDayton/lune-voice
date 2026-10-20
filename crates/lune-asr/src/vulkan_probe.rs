// SPDX-License-Identifier: GPL-3.0-or-later

use std::ffi::{c_char, c_void};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const REPROBE_AFTER: Duration = Duration::from_secs(10);

pub(crate) const VK_API_VERSION_1_2: u32 = (1 << 22) | (2 << 12);
const VK_API_VARIANT_MASK: u32 = 0x7 << 29;
const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: i32 = 2;
const VK_SUCCESS: i32 = 0;
const VK_INCOMPLETE: i32 = 5;
const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const MAX_DEVICES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhysicalDevice {
    pub api_version: u32,
    pub device_type: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    NoDriver,
    OutdatedDriver,
    NoDiscreteGpu,
    Ready,
}

pub(crate) fn classify(devices: Option<&[PhysicalDevice]>) -> Probe {
    let Some(devices) = devices.filter(|d| !d.is_empty()) else {
        return Probe::NoDriver;
    };
    let discrete: Vec<_> = devices
        .iter()
        .filter(|d| d.device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        .collect();
    if discrete.is_empty() {
        return Probe::NoDiscreteGpu;
    }
    if discrete
        .iter()
        .any(|d| d.api_version & !VK_API_VARIANT_MASK >= VK_API_VERSION_1_2)
    {
        Probe::Ready
    } else {
        Probe::OutdatedDriver
    }
}

fn reusable(last: Option<(Probe, Instant)>, now: Instant) -> Option<Probe> {
    let (probe, at) = last?;
    (probe == Probe::Ready || now.duration_since(at) < REPROBE_AFTER).then_some(probe)
}

pub fn probe() -> Probe {
    static LAST: Mutex<Option<(Probe, Instant)>> = Mutex::new(None);
    let mut last = LAST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(probe) = reusable(*last, Instant::now()) {
        return probe;
    }
    let devices = enumerate();
    let result = classify(devices.as_deref());
    tracing::info!(?devices, ?result, "Vulkan GPU probe");
    *last = Some((result, Instant::now()));
    result
}

type VoidFn = unsafe extern "system" fn();
type GetInstanceProcAddr = unsafe extern "system" fn(*mut c_void, *const c_char) -> Option<VoidFn>;
type CreateInstance =
    unsafe extern "system" fn(*const InstanceCreateInfo, *const c_void, *mut *mut c_void) -> i32;
type DestroyInstance = unsafe extern "system" fn(*mut c_void, *const c_void);
type EnumeratePhysicalDevices =
    unsafe extern "system" fn(*mut c_void, *mut u32, *mut *mut c_void) -> i32;
type GetPhysicalDeviceProperties = unsafe extern "system" fn(*mut c_void, *mut PropertiesBuf);

#[repr(C)]
struct ApplicationInfo {
    s_type: u32,
    p_next: *const c_void,
    p_application_name: *const c_char,
    application_version: u32,
    p_engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct InstanceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_application_info: *const ApplicationInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
}

#[repr(C, align(8))]
struct PropertiesBuf([u8; 1024]);

fn enumerate() -> Option<Vec<PhysicalDevice>> {
    let gipa = loader::get_instance_proc_addr()?;
    // SAFETY: every pointer handed to Vulkan points at a live, correctly laid
    // out `repr(C)` value for the duration of the call; function pointers come
    // from the loader's own `vkGetInstanceProcAddr` and are transmuted to their
    // documented signatures; the instance is destroyed before returning.
    unsafe {
        let create: CreateInstance =
            std::mem::transmute(gipa(std::ptr::null_mut(), c"vkCreateInstance".as_ptr())?);
        let app = ApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: c"lune-voice".as_ptr(),
            application_version: 0,
            p_engine_name: std::ptr::null(),
            engine_version: 0,
            api_version: VK_API_VERSION_1_2,
        };
        let info = InstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            p_application_info: &app,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
        };
        let mut instance: *mut c_void = std::ptr::null_mut();
        if create(&info, std::ptr::null(), &mut instance) != VK_SUCCESS || instance.is_null() {
            return None;
        }
        let devices = list_devices(gipa, instance);
        if let Some(destroy) = gipa(instance, c"vkDestroyInstance".as_ptr()) {
            let destroy: DestroyInstance = std::mem::transmute(destroy);
            destroy(instance, std::ptr::null());
        }
        devices
    }
}

unsafe fn list_devices(
    gipa: GetInstanceProcAddr,
    instance: *mut c_void,
) -> Option<Vec<PhysicalDevice>> {
    // SAFETY: see `enumerate`; the device array holds `count` slots.
    unsafe {
        let enumerate: EnumeratePhysicalDevices =
            std::mem::transmute(gipa(instance, c"vkEnumeratePhysicalDevices".as_ptr())?);
        let properties: GetPhysicalDeviceProperties =
            std::mem::transmute(gipa(instance, c"vkGetPhysicalDeviceProperties".as_ptr())?);
        let mut handles = [std::ptr::null_mut::<c_void>(); MAX_DEVICES];
        let mut count = MAX_DEVICES as u32;
        let status = enumerate(instance, &mut count, handles.as_mut_ptr());
        if status != VK_SUCCESS && status != VK_INCOMPLETE {
            return None;
        }
        let found = handles[..(count as usize).min(MAX_DEVICES)]
            .iter()
            .filter(|h| !h.is_null())
            .map(|&handle| {
                let mut buf = PropertiesBuf([0; 1024]);
                properties(handle, &mut buf);
                let word = |at: usize| [buf.0[at], buf.0[at + 1], buf.0[at + 2], buf.0[at + 3]];
                PhysicalDevice {
                    api_version: u32::from_ne_bytes(word(0)),
                    device_type: i32::from_ne_bytes(word(16)),
                }
            })
            .collect();
        Some(found)
    }
}

#[cfg(unix)]
mod loader {
    use super::GetInstanceProcAddr;

    pub(super) fn get_instance_proc_addr() -> Option<GetInstanceProcAddr> {
        // SAFETY: both strings are NUL-terminated literals; a non-null symbol
        // from the Vulkan loader has the `vkGetInstanceProcAddr` signature.
        unsafe {
            let handle = libc::dlopen(
                c"libvulkan.so.1".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            );
            if handle.is_null() {
                return None;
            }
            let sym = libc::dlsym(handle, c"vkGetInstanceProcAddr".as_ptr());
            if sym.is_null() {
                return None;
            }
            Some(std::mem::transmute::<*mut libc::c_void, GetInstanceProcAddr>(sym))
        }
    }
}

#[cfg(target_os = "windows")]
mod loader {
    use super::GetInstanceProcAddr;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
    };

    pub(super) fn get_instance_proc_addr() -> Option<GetInstanceProcAddr> {
        let name: Vec<u16> = "vulkan-1.dll"
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `name` and the symbol name are NUL-terminated and outlive
        // the calls; the exported symbol has the documented signature.
        unsafe {
            let module = LoadLibraryExW(
                name.as_ptr(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_SYSTEM32,
            );
            if module.is_null() {
                return None;
            }
            let sym = GetProcAddress(module, c"vkGetInstanceProcAddr".as_ptr().cast())?;
            Some(std::mem::transmute::<
                unsafe extern "system" fn() -> isize,
                GetInstanceProcAddr,
            >(sym))
        }
    }
}

#[cfg(not(any(unix, target_os = "windows")))]
mod loader {
    use super::GetInstanceProcAddr;

    pub(super) fn get_instance_proc_addr() -> Option<GetInstanceProcAddr> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(api_version: u32, device_type: i32) -> PhysicalDevice {
        PhysicalDevice {
            api_version,
            device_type,
        }
    }

    const V1_1: u32 = (1 << 22) | (1 << 12);
    const V1_3: u32 = (1 << 22) | (3 << 12) | 250;
    const INTEGRATED: i32 = 1;
    const DISCRETE: i32 = 2;
    const CPU: i32 = 4;

    #[test]
    fn no_loader_or_no_device_means_no_driver() {
        assert_eq!(classify(None), Probe::NoDriver);
        assert_eq!(classify(Some(&[])), Probe::NoDriver);
    }

    #[test]
    fn a_discrete_gpu_on_vulkan_1_2_or_newer_qualifies() {
        assert_eq!(classify(Some(&[gpu(V1_3, DISCRETE)])), Probe::Ready);
        assert_eq!(
            classify(Some(&[gpu(VK_API_VERSION_1_2, DISCRETE)])),
            Probe::Ready
        );
        assert_eq!(
            classify(Some(&[gpu(V1_3, INTEGRATED), gpu(V1_3, DISCRETE)])),
            Probe::Ready
        );
    }

    #[test]
    fn a_discrete_gpu_on_an_older_vulkan_needs_a_driver_update() {
        assert_eq!(
            classify(Some(&[gpu(V1_1, DISCRETE)])),
            Probe::OutdatedDriver
        );
    }

    #[test]
    fn a_found_gpu_is_kept_and_anything_else_is_asked_again_after_a_while() {
        let start = Instant::now();
        let later = start + REPROBE_AFTER;
        assert_eq!(reusable(None, start), None);
        assert_eq!(
            reusable(Some((Probe::Ready, start)), later),
            Some(Probe::Ready)
        );
        for miss in [Probe::NoDriver, Probe::OutdatedDriver, Probe::NoDiscreteGpu] {
            assert_eq!(reusable(Some((miss, start)), start), Some(miss));
            assert_eq!(reusable(Some((miss, start)), later), None);
        }
    }

    #[test]
    fn integrated_and_software_devices_alone_do_not_qualify() {
        assert_eq!(
            classify(Some(&[gpu(V1_3, INTEGRATED), gpu(V1_3, CPU)])),
            Probe::NoDiscreteGpu
        );
    }

    #[test]
    fn the_api_variant_bits_do_not_count_as_a_newer_version() {
        let variant_1_1 = (1 << 29) | V1_1;
        assert_eq!(
            classify(Some(&[gpu(variant_1_1, DISCRETE)])),
            Probe::OutdatedDriver
        );
    }
}
