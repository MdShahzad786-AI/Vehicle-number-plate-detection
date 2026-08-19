"""
Ye script automatically test images download kar degi
"""

import urllib.request
import os

# Test images URLs (free, no copyright)
test_images = {
    'car1.jpg': 'https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=800',
    'car2.jpg': 'https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=800',
    'car3.jpg': 'https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=800',
    'car4.jpg': 'https://images.unsplash.com/photo-1605559424843-9e4c228bf1c2?w=800',
}

def download_images():
    """
    Test images download karo
    """
    print("📥 Downloading test images...")
    
    # data/images folder banao agar nahi hai
    os.makedirs('data/images', exist_ok=True)
    
    for filename, url in test_images.items():
        output_path = f'data/images/{filename}'
        
        if os.path.exists(output_path):
            print(f"✅ {filename} already exists, skipping...")
            continue
        
        try:
            print(f"⏬ Downloading {filename}...")
            urllib.request.urlretrieve(url, output_path)
            print(f"✅ {filename} downloaded successfully!")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
    
    print("\n🎉 All images downloaded!")
    print(f"📁 Location: data/images/")

if __name__ == "__main__":
    download_images()