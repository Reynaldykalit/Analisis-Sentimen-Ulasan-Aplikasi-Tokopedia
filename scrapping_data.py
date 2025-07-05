from google_play_scraper import reviews_all, Sort
import csv
import time

def scrape_tokopedia():
    try:
        print("Memulai scraping ulasan Tokopedia (maks 40000)...")
        
        # Mendapatkan semua review
        print("Mengambil data dari Google Play Store...")
        all_reviews = reviews_all(
            'com.tokopedia.tkpd',
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            sleep_milliseconds=2000
        )
        
        # Batasi data ke 40000 ulasan
        limited_reviews = all_reviews[:40000]
        total_reviews = len(limited_reviews)
        
        print(f"Berhasil mendapatkan {total_reviews} ulasan")
        print("Menyimpan data ke CSV...")
        
        with open('ulasan_tokopedia.csv', mode='w', encoding='utf-8-sig', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow(['reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 
                           'reviewDate', 'replyContent', 'replyDate', 'appVersion'])
            
            # Variabel untuk melacak progres
            progress_step = max(1, total_reviews // 100)  # Update setiap 1%
            start_time = time.time()
            
            for i, review in enumerate(limited_reviews):
                writer.writerow([
                    review.get('reviewId', ''),
                    review.get('userName', 'Anonim'),
                    review.get('content', ''),
                    review.get('score', 0),
                    review.get('thumbsUpCount', 0), 
                    review.get('at', '').strftime('%Y-%m-%d %H:%M:%S') if review.get('at') else '',
                    review.get('replyContent', ''),
                    review.get('repliedAt', '').strftime('%Y-%m-%d %H:%M:%S') if review.get('repliedAt') else '',
                    review.get('appVersion', '')
                ])
        
        print("Data berhasil disimpan ke ulasan_tokopedia.csv")
        print(f"Total waktu eksekusi: {time.time() - start_time:.2f} detik")
        
    except Exception as e:
        print(f"Gagal scraping: {str(e)}")

if __name__ == "__main__":
    scrape_tokopedia()