import mysql.connector
from mysql.connector import Error

def create_connection():
    """Tạo kết nối đến cơ sở dữ liệu MySQL"""
    try:
        conn = mysql.connector.connect(
            host="localhost",  # Địa chỉ máy chủ MySQL
            user="root",       # Tên người dùng MySQL
            password="123456",  # Mật khẩu người dùng MySQL
            database="pbl6"    # Tên cơ sở dữ liệu MySQL
        )
        return conn
    except Error as e:
        print(f"Lỗi khi tạo kết nối: {e}")
        return None


def check_connection():
    """Kiểm tra kết nối cơ sở dữ liệu"""
    conn = create_connection()
    if conn:
        try:
            if conn.is_connected():
                print("Kết nối cơ sở dữ liệu thành công!")
        except Error as e:
            print(f"Lỗi khi kiểm tra kết nối: {e}")
        finally:
            conn.close()
            print("Đã đóng kết nối cơ sở dữ liệu.")
    else:
        print("Không thể tạo kết nối đến cơ sở dữ liệu.")

def update_record(table, column, new_value, condition):
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Tạo câu lệnh SQL UPDATE
            sql_update_query = f"UPDATE {table} SET {column} = %s WHERE {condition}"
            # Thực thi câu lệnh
            cursor.execute(sql_update_query, (new_value,))
            conn.commit()  # Lưu thay đổi
            print(f"Cập nhật thành công: {cursor.rowcount} bản ghi bị ảnh hưởng.")
        except Error as e:
            print(f"Lỗi khi cập nhật bản ghi: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
                print("Đã đóng kết nối cơ sở dữ liệu.")
    else:
        print("Không thể kết nối đến cơ sở dữ liệu.")

def get_music_title_by_id(music_id):
    """Kiểm tra xem bài hát có tồn tại theo ID hay không"""
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT title FROM musics WHERE music_id = %s", (music_id,))
            music = cursor.fetchone()
            return music
        except Error as e:
            print(f"Lỗi truy vấn: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

def update_music_title_by_id(music_id, new_title):
    """Cập nhật tiêu đề bài hát theo ID"""
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            sql_update_query = "UPDATE musics SET title = %s WHERE music_id = %s"
            cursor.execute(sql_update_query, (new_title, music_id))
            conn.commit()
            print(f"Cập nhật thành công: {cursor.rowcount} bản ghi bị ảnh hưởng.")
        except Error as e:
            print(f"Lỗi khi cập nhật bản ghi: {e}")
        finally:
            cursor.close()
            conn.close()

def update_music_url_by_id(music_id, new_url):
    """Cập nhật url bài hát theo ID"""
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            sql_update_query = "UPDATE musics SET full_url = %s WHERE music_id = %s"
            cursor.execute(sql_update_query, (new_url, music_id))
            conn.commit()
            print(f"Cập nhật thành công: {cursor.rowcount} bản ghi bị ảnh hưởng.")
        except Error as e:
            print(f"Lỗi khi cập nhật bản ghi: {e}")
        finally:
            cursor.close()
            conn.close()

def update_music_isPurchased_by_id(music_id):
    """Cập nhật trang thái mua hàng bài hát theo ID"""
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            sql_update_query = "UPDATE musics SET is_purchased = TRUE WHERE music_id = %s"
            cursor.execute(sql_update_query, (music_id,))
            conn.commit()
            print(f"Cập nhật thành công: {cursor.rowcount} bản ghi bị ảnh hưởng.")
        except Error as e:
            print(f"Lỗi khi cập nhật bản ghi: {e}")
        finally:
            cursor.close()
            conn.close()