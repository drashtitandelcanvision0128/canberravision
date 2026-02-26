"""
Modular App Integration for YOLO26
Clean integration of image, video, and webcam processors
Add this to your main app.py
"""

# ================================================================
# ADD THESE IMPORTS TO YOUR app.py (after existing imports)
# ================================================================

"""
# Add these imports after your existing imports (around line 70-80)

# Import modular processors
try:
    from image_processor import process_single_image, get_image_processing_stats
    from video_processor import process_single_video, process_video_in_chunks, get_video_processing_stats
    from webcam_processor import start_webcam, stop_webcam, get_webcam_frame, get_webcam_stats
    from video_output_handler import get_outputs_info, cleanup_outputs
    
    MODULAR_PROCESSORS_AVAILABLE = True
    print("[INFO] Modular processors loaded successfully")
except ImportError as e:
    print(f"[WARNING] Modular processors not available: {e}")
    MODULAR_PROCESSORS_AVAILABLE = False
"""

# ================================================================
# REPLACE YOUR MAIN PROCESSING FUNCTIONS WITH THESE
# ================================================================

def process_image_modular(image_input, use_enhanced=True, use_international=True):
    """
    Modular image processing function
    Replaces your existing image processing
    """
    try:
        if not MODULAR_PROCESSORS_AVAILABLE:
            return {"error": "Modular processors not available"}
        
        # Convert input to numpy array if needed
        if hasattr(image_input, 'read'):  # File-like object
            import numpy as np
            image_bytes = image_input.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            image = image_input
        
        if image is None:
            return {"error": "Invalid image input"}
        
        # Process using modular image processor
        result = process_single_image(
            image, 
            use_enhanced=use_enhanced, 
            use_international=use_international
        )
        
        # Add processing stats
        stats = get_image_processing_stats()
        result['processing_stats'] = stats
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Modular image processing failed: {e}")
        return {"error": str(e)}

def process_video_modular(video_input, save_output=True, progress_callback=None):
    """
    Modular video processing function
    Replaces your existing video processing
    """
    try:
        if not MODULAR_PROCESSORS_AVAILABLE:
            return {"error": "Modular processors not available"}
        
        # Get video path
        if hasattr(video_input, 'name'):  # Uploaded file
            video_path = video_input.name
        elif isinstance(video_input, str):  # Path string
            video_path = video_input
        else:
            return {"error": "Invalid video input"}
        
        # Process using modular video processor
        result = process_single_video(
            video_path, 
            save_output=save_output, 
            progress_callback=progress_callback
        )
        
        # Add processing stats
        stats = get_video_processing_stats()
        result['processing_stats'] = stats
        
        # Add output info
        if save_output:
            output_info = get_outputs_info()
            result['output_info'] = output_info
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Modular video processing failed: {e}")
        return {"error": str(e)}

def process_webcam_modular(camera_index=0, duration_seconds=None):
    """
    Modular webcam processing function
    Replaces your existing webcam processing
    """
    try:
        if not MODULAR_PROCESSORS_AVAILABLE:
            return {"error": "Modular processors not available"}
        
        # Start webcam
        start_result = start_webcam(camera_index)
        if 'error' in start_result:
            return start_result
        
        print(f"[INFO] Webcam started: {start_result}")
        
        frames_processed = []
        start_time = time.time()
        
        try:
            # Process frames
            while True:
                frame = get_webcam_frame()
                if frame is not None:
                    frames_processed.append(frame.copy())
                
                # Check duration limit
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("[INFO] Webcam processing interrupted")
        
        # Stop webcam
        stop_result = stop_webcam()
        
        # Get final stats
        stats = get_webcam_stats()
        
        return {
            'success': True,
            'frames_processed': len(frames_processed),
            'processing_stats': stats,
            'start_result': start_result,
            'stop_result': stop_result,
            'sample_frames': frames_processed[:5]  # Return first 5 frames as sample
        }
        
    except Exception as e:
        print(f"[ERROR] Modular webcam processing failed: {e}")
        return {"error": str(e)}

# ================================================================
# GRADIO INTERFACE UPDATES
# ================================================================

def create_modular_gradio_interface():
    """
    Create updated Gradio interface with modular processors
    Replace your existing Gradio functions with these
    """
    
    def process_image_gradio(image, enhanced_detection, international_recognition):
        """Gradio image processing function"""
        try:
            if image is None:
                return None, "Please upload an image"
            
            result = process_image_modular(
                image, 
                use_enhanced=enhanced_detection, 
                use_international=international_recognition
            )
            
            if 'error' in result:
                return None, f"❌ Error: {result['error']}"
            
            # Create annotated image
            annotated_image = result.get('output_files', {}).get('detection_image')
            
            # Format results for display
            display_text = format_results_for_display(result)
            
            return annotated_image, display_text
            
        except Exception as e:
            return None, f"❌ Processing failed: {str(e)}"
    
    def process_video_gradio(video, save_output):
        """Gradio video processing function"""
        try:
            if video is None:
                return None, "Please upload a video"
            
            # Progress callback
            def progress_callback(progress, current, total):
                print(f"Progress: {progress:.1f}% ({current}/{total})")
            
            result = process_video_modular(
                video, 
                save_output=save_output, 
                progress_callback=progress_callback
            )
            
            if 'error' in result:
                return None, f"❌ Error: {result['error']}"
            
            # Get output video path
            output_video = result.get('output_files', {}).get('saved_video')
            
            # Format results
            display_text = format_video_results_for_display(result)
            
            return output_video, display_text
            
        except Exception as e:
            return None, f"❌ Processing failed: {str(e)}"
    
    def process_webcam_gradio(camera_index, duration):
        """Gradio webcam processing function"""
        try:
            result = process_webcam_moderal(
                camera_index=int(camera_index), 
                duration_seconds=int(duration) if duration else None
            )
            
            if 'error' in result:
                return None, f"❌ Error: {result['error']}"
            
            # Format results
            display_text = format_webcam_results_for_display(result)
            
            # Return sample frame if available
            sample_frame = None
            if result.get('sample_frames'):
                sample_frame = result['sample_frames'][0]
            
            return sample_frame, display_text
            
        except Exception as e:
            return None, f"❌ Processing failed: {str(e)}"
    
    return {
        'image_fn': process_image_gradio,
        'video_fn': process_video_gradio,
        'webcam_fn': process_webcam_gradio
    }

# ================================================================
# RESULT FORMATTING FUNCTIONS
# ================================================================

def format_results_for_display(result):
    """Format image processing results for display"""
    try:
        if 'error' in result:
            return f"❌ Error: {result['error']}"
        
        display_text = "🚗 YOLO26 Detection Results\n"
        display_text += "=" * 30 + "\n\n"
        
        # Image info
        image_info = result.get('image_info', {})
        display_text += f"📷 Image Info:\n"
        display_text += f"   Size: {image_info.get('shape', 'Unknown')}\n"
        display_text += f"   Processed: {image_info.get('timestamp', 'Unknown')}\n\n"
        
        # Detections
        detections = result.get('detections', {})
        
        # License plates
        plates = detections.get('license_plates', [])
        display_text += f"🔢 License Plates ({len(plates)} found):\n"
        for i, plate in enumerate(plates[:5]):  # Show first 5
            display_text += f"   {i+1}. {plate.get('plate_text', 'Unknown')} "
            display_text += f"({plate.get('confidence', 0):.2f})\n"
        
        if len(plates) > 5:
            display_text += f"   ... and {len(plates) - 5} more\n"
        
        # Objects
        objects = detections.get('objects', [])
        display_text += f"\n🚗 Objects ({len(objects)} found):\n"
        object_counts = {}
        for obj in objects:
            class_name = obj.get('class_name', 'Unknown')
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        for class_name, count in object_counts.items():
            display_text += f"   • {class_name}: {count}\n"
        
        # International plates
        if 'international_plates' in result:
            intl_plates = result['international_plates'].get('plates', [])
            if intl_plates:
                display_text += f"\n🌍 International Recognition:\n"
                for plate in intl_plates[:3]:  # Show first 3
                    countries = plate.get('countries', [])
                    if countries:
                        top_country = countries[0]
                        display_text += f"   • {plate.get('text', 'Unknown')}: "
                        display_text += f"{top_country.get('country', 'Unknown')}\n"
        
        # Processing info
        proc_info = result.get('processing_info', {})
        display_text += f"\n⚙️ Processing Info:\n"
        display_text += f"   Time: {proc_info.get('processing_time', 0):.2f}s\n"
        display_text += f"   Methods: {', '.join(proc_info.get('method_used', []))}\n"
        
        # Output files
        output_files = result.get('output_files', {})
        saved_frames = output_files.get('saved_frames', [])
        if saved_frames:
            display_text += f"\n💾 Saved Files:\n"
            display_text += f"   Frames: {len(saved_frames)} snapshots saved\n"
        
        return display_text
        
    except Exception as e:
        return f"❌ Error formatting results: {str(e)}"

def format_video_results_for_display(result):
    """Format video processing results for display"""
    try:
        if 'error' in result:
            return f"❌ Error: {result['error']}"
        
        display_text = "📹 YOLO26 Video Processing Results\n"
        display_text += "=" * 35 + "\n\n"
        
        # Video info
        video_info = result.get('video_info', {})
        display_text += f"📹 Video Info:\n"
        display_text += f"   Resolution: {video_info.get('width', 0)}x{video_info.get('height', 0)}\n"
        display_text += f"   Duration: {video_info.get('duration', 0):.1f}s\n"
        display_text += f"   Total Frames: {video_info.get('frame_count', 0)}\n\n"
        
        # Processing summary
        proc_info = result.get('processing_info', {})
        display_text += f"⚙️ Processing Summary:\n"
        display_text += f"   Frames Processed: {proc_info.get('processed_frames', 0)}\n"
        display_text += f"   Frames with Detections: {proc_info.get('frames_with_detections', 0)}\n"
        display_text += f"   Processing Time: {proc_info.get('processing_time', 0):.1f}s\n"
        display_text += f"   Processing FPS: {proc_info.get('fps_processed', 0):.1f}\n\n"
        
        # License plates summary
        plate_summary = result.get('detections', {}).get('summary', {})
        unique_plates = plate_summary.get('unique_plates', [])
        display_text += f"🔢 License Plates Summary:\n"
        display_text += f"   Total Detections: {plate_summary.get('total_license_plates', 0)}\n"
        display_text += f"   Unique Plates: {len(unique_plates)}\n"
        
        if unique_plates:
            display_text += f"   Plates Found: {', '.join(unique_plates[:5])}\n"
            if len(unique_plates) > 5:
                display_text += f"   ... and {len(unique_plates) - 5} more\n"
        
        # Output files
        output_files = result.get('output_files', {})
        display_text += f"\n💾 Output Files:\n"
        
        saved_video = output_files.get('saved_video')
        if saved_video:
            display_text += f"   📹 Processed Video: {saved_video}\n"
        
        saved_frames = output_files.get('saved_frames', [])
        if saved_frames:
            display_text += f"   🖼️  Detection Frames: {len(saved_frames)} saved\n"
        
        # Output info
        output_info = result.get('output_info', {})
        if output_info:
            display_text += f"\n📊 Output Directory Info:\n"
            display_text += f"   Total Files: {output_info.get('total_files', 0)}\n"
            display_text += f"   Total Size: {output_info.get('total_size_mb', 0):.1f} MB\n"
        
        return display_text
        
    except Exception as e:
        return f"❌ Error formatting video results: {str(e)}"

def format_webcam_results_for_display(result):
    """Format webcam processing results for display"""
    try:
        if 'error' in result:
            return f"❌ Error: {result['error']}"
        
        display_text = "📷 YOLO26 Webcam Processing Results\n"
        display_text += "=" * 35 + "\n\n"
        
        # Session info
        display_text += f"📷 Session Info:\n"
        display_text += f"   Frames Processed: {result.get('frames_processed', 0)}\n"
        
        # Processing stats
        stats = result.get('processing_stats', {})
        display_text += f"\n⚙️ Performance Stats:\n"
        display_text += f"   Current FPS: {stats.get('current_fps', 0):.1f}\n"
        display_text += f"   Runtime: {stats.get('runtime_seconds', 0):.1f}s\n"
        display_text += f"   Avg Processing Time: {stats.get('avg_processing_time_ms', 0):.1f}ms\n"
        
        # Detection results
        display_text += f"\n🔢 Detection Results:\n"
        display_text += f"   Unique Plates Found: {stats.get('unique_plates_count', 0)}\n"
        
        unique_plates = stats.get('unique_plates_found', [])
        if unique_plates:
            display_text += f"   Plates: {', '.join(unique_plates[:5])}\n"
            if len(unique_plates) > 5:
                display_text += f"   ... and {len(unique_plates) - 5} more\n"
        
        # Camera info
        start_result = result.get('start_result', {})
        if start_result:
            display_text += f"\n📹 Camera Info:\n"
            display_text += f"   Camera Index: {start_result.get('camera_index', 'Unknown')}\n"
            display_text += f"   Resolution: {start_result.get('resolution', 'Unknown')}\n"
            display_text += f"   FPS: {start_result.get('fps', 'Unknown')}\n"
        
        return display_text
        
    except Exception as e:
        return f"❌ Error formatting webcam results: {str(e)}"

# ================================================================
# CLEANUP AND MAINTENANCE
# ================================================================

def setup_modular_system():
    """
    Setup and initialize modular system
    Call this when your app starts
    """
    try:
        if MODULAR_PROCESSORS_AVAILABLE:
            print("[INFO] Setting up modular YOLO26 system...")
            
            # Cleanup old outputs
            cleanup_outputs(days_old=7)
            
            # Get system info
            image_stats = get_image_processing_stats()
            video_stats = get_video_processing_stats()
            
            print("[INFO] Modular system ready!")
            print(f"   Image processor: {image_stats.get('total_processed', 0)} images processed")
            print(f"   Video processor: {video_stats.get('total_videos_processed', 0)} videos processed")
            
            return True
        else:
            print("[WARNING] Modular processors not available")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to setup modular system: {e}")
        return False

# ================================================================
# USAGE EXAMPLE
# ================================================================

if __name__ == "__main__":
    print("🔧 Modular App Integration Guide")
    print("=" * 40)
    
    print("📋 Integration Steps:")
    print("1. Add imports to your app.py")
    print("2. Replace processing functions with modular versions")
    print("3. Update Gradio interface functions")
    print("4. Call setup_modular_system() when app starts")
    
    print("\n🎯 Benefits:")
    print("✅ Clean, modular code structure")
    print("✅ Separate files for each media type")
    print("✅ Easy maintenance and debugging")
    print("✅ Reusable components")
    print("✅ Better error handling")
    print("✅ Performance optimization")
    
    print("\n📁 File Structure:")
    print("app.py                 # Main application (clean)")
    print("├── image_processor.py  # Image processing logic")
    print("├── video_processor.py  # Video processing logic")
    print("├── webcam_processor.py # Webcam processing logic")
    print("└── video_output_handler.py # Output management")
    
    print("\n✅ Modular integration ready!")
    print("🚀 Your app.py will be much cleaner and organized!")
