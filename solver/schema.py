"""Mapping of raw columns to canonical fields.

Columns are mapped first through the README description of the column (the
READMEs use a fixed vocabulary), then through name aliases for files whose
README gives no column list.
"""
import re

from textnorm import norm

# (regex on normalised description, field). First match wins, so specific
# phrasings come before generic ones.
DESC_RULES = [
    (r'^cot trong', 'EMPTY'),
    (r'^so thu tu dong trong phieu nhap', 'rc_line'),
    (r'^so thu tu dong trong don', 'line_no'),
    (r'^so thu tu don', 'order_id'),
    (r'^so hoa don', 'order_id'),
    (r'^ma don', 'order_id'),
    (r'^ma phieu nhap', 'rc_id'),
    (r'^ma phieu thu', 'pay_id'),
    (r'^ma nha cung cap', 'supplier_id'),
    (r'^ma nhan vien', 'staff_id'),
    (r'^ma khach', 'cust_id'),
    (r'^ma hang', 'prod_id'),
    (r'^so lo', 'lot_id'),
    (r'^ten hang file chi co', 'prod_name'),
    (r'^ten hang', 'prod_name'),
    (r'^hang ban', 'prod_name'),
    (r'^ten khach', 'cust_name'),
    (r'^ten goi khac', 'alias_names'),
    (r'^ten o nha cua be', 'baby_name'),
    (r'^ten chuong trinh', 'promo_name'),
    (r'^ten$', 'name'),
    (r'^nhom hang', 'category'),
    (r'^nhom khach', 'cust_group'),
    (r'^so luong da doi ra don vi co ban', 'qty_base'),
    (r'^so luong ban lay tu lo', 'lot_qty'),
    (r'^so luong khach tra lai', 'lot_ret_qty'),
    (r'^so luong nhap don vi co ban', 'rc_qty_base'),
    (r'^so luong nhap', 'rc_qty'),
    (r'^so luong', 'qty'),
    (r'^don vi co ban', 'base_unit'),
    (r'^don vi lon', 'big_unit'),
    (r'^don vi nhap', 'rc_unit'),
    (r'^don vi tinh', 'unit'),
    (r'^quy doi don vi lon', 'unit_rule'),
    (r'^he so', 'factor'),
    (r'^ngay ban', 'date'),
    (r'^ngay sua', 'date'),
    (r'^ngay nhap', 'rc_date'),
    (r'^ngay giao', 'deliv_date'),
    (r'^ngay bat dau ap dung gia', 'price_date'),
    (r'^ngay tao mat hang', 'prod_created'),
    (r'^ngay tao khach', 'cust_created'),
    (r'^ngay dang ky', 'cust_created'),
    (r'^ngay vao lam', 'start_date'),
    (r'^tu ngay', 'from_date'),
    (r'^den ngay', 'to_date'),
    (r'^thoi diem dat don', 'datetime'),
    (r'^gio dat don', 'time'),
    (r'^thoi diem thu', 'paid_at'),
    (r'^han su dung', 'expiry'),
    (r'^han thanh toan', 'due_date'),
    (r'^so ngay duoc no', 'credit_days'),
    (r'^trang thai don', 'status'),
    (r'^trang thai phieu nhap', 'rc_status'),
    (r'^don gia', 'price'),
    (r'^gia ban hien tai', 'cur_price'),
    (r'^gia ban ap dung', 'new_price'),
    (r'^gia truoc lan doi', 'old_price'),
    (r'^gia nhap', 'rc_price'),
    (r'^gia von', 'cost'),
    (r'^thanh tien', 'amount'),
    (r'^tien nhap cua ca dong', 'rc_amount'),
    (r'^tong tien phieu nhap', 'rc_total'),
    (r'^tong tien cua ca phieu', 'payable'),
    (r'^tien giam cua ca dong', 'line_disc'),
    (r'^giam gia cua ca don', 'order_disc'),
    (r'^tien bot cho ca don', 'order_disc'),
    (r'^giam gia', 'discount'),
    (r'^tong tien hang cua', 'goods_total'),
    (r'^tien hang cua don', 'goods_total'),
    (r'^tong tien don', 'payable'),
    (r'^so khach phai tra', 'payable'),
    (r'^khach can tra', 'payable'),
    (r'^so tien khach tra cho ca don', 'cust_paid_net'),
    (r'^so khach con no', 'debt'),
    (r'^khach con no', 'debt'),
    (r'^con no', 'debt'),
    (r'^khach da tra', 'paid'),
    (r'^da thanh toan', 'paid'),
    (r'^phi giao hang', 'ship_fee'),
    (r'^tien thu may cu', 'tradein'),
    (r'^hinh thuc thanh toan', 'pay_method'),
    (r'^so tien', 'pay_amount'),
    (r'^kenh ban', 'channel'),
    (r'^so dien thoai', 'phone'),
    (r'^dia chi', 'address'),
    (r'^ghi chu', 'note'),
    (r'^ban$', 'table'),
    (r'^khu vuc', 'area'),
    (r'^so khach cua don', 'guests'),
    (r'^dot giao', 'deliv_batch'),
    (r'^bien so xe', 'plate'),
    (r'^tai xe', 'driver'),
    (r'^hoat chat', 'ingredient'),
    (r'^thuoc ke don', 'rx'),
    (r'^nha cung cap', 'supplier'),
    (r'^con kinh doanh', 'active'),
    (r'^ton kho', 'stock'),
    (r'^imei', 'imei'),
    (r'^hang$', 'brand'),
    (r'^so cua sua', 'size'),
    (r'^diem con', 'points'),
    (r'^hang ap dung', 'promo_target'),
    (r'^dong xe va bien so', 'vehicle'),
    (r'^cac viec da lam', 'jobs'),
    (r'^tho lam phieu', 'mechanic'),
    (r'^tong tien ca phieu', 'amount'),
    (r'^ho ten tho', 'name'),
    (r'^phan tram hoa hong', 'commission'),
]
DESC_RULES = [(re.compile(p), f) for p, f in DESC_RULES]

# Name aliases for files without a column list in the README. Keys are
# normalised column names.
NAME_ALIASES = {
    # description-less exports (pharmacy / cafe multi-file exports)
    'order id': 'order_id', 'ma giao dich': 'order_id', 'chi tiet hang': 'prod_name', 'slg': 'qty',
    'sl co ban': 'qty_base', 'khuyen mai': 'line_disc', 'gia niem yet': 'price', 'tt dong': 'line_no',
    'd vi': 'unit', 'ten kh': 'cust_name', 'nguon don': 'channel', 'ma kh mua': 'cust_id', 'tong tien': 'payable',
    'ck': 'discount', 'lo': 'lot_id', 'sl quy ra le': 'lot_qty', 'sl tra dvt co ban': 'lot_ret_qty', 'no': 'line_no',
    'don vi': 'unit', 'gia ban theo don vi quy doi': 'unit_price_big', 'ma sp': 'prod_id', 'phan loai': 'category',
    'mon': 'prod_name', 'gia': 'cur_price', 'customer id': 'cust_id', 'so dien thoai': 'phone',
    'hang': 'cust_group', 'kieu khach': 'cust_group', 'khach hang mua': 'cust_name', 'dia chi': 'address',
    'ma san pham': 'prod_id', 'gia moi': 'new_price', 'ngay ap dung': 'price_date', 'san pham': 'prod_name',
    'gia cu': 'old_price', 'item': 'prod_name', 'hsd': 'expiry', 'ma mon': 'prod_id',
    'sl nhap dvt co ban': 'rc_qty_base', 'ma phieu nhap': 'rc_id', 'phieu thu': 'pay_id', 'thoi gian thu': 'paid_at',
    'ma hh': 'prod_id', 'hoat chat': 'ingredient', 'ke don': 'rx', 'sl nhap': 'rc_qty', 'mo ta': 'note',
    'product': 'prod_name', 'ngay nhap': 'rc_date', 'dvt nhap': 'rc_unit', 'id sp': 'prod_id',
    'ten hang': 'prod_name', 'tien thu': 'pay_amount', 'ma vach': 'prod_id', 'luu y': 'note', 'ma khach hang': 'cust_id',
    'ten khach hang': 'cust_name', 'dien thoai': 'phone', 'nhom khach hang': 'cust_group', 'ngay tao': 'cust_created',
    'giam d': 'line_disc', 's l': 'qty', 'sl dv co ban': 'qty_base', 'trang thai giao': 'status',
    'nhom hang hoa': 'category', 'ma hang hoa': 'prod_id', 'nhom': 'category', 'loai hang': 'category',
    # orders / invoices
    'ma hoa don': 'order_id', 'ma don': 'order_id', 'ma don hang': 'order_id', 'don so': 'order_id',
    'so ct': 'order_id', 'ma bill': 'order_id', 'bill': 'order_id', 'mahoadon': 'order_id',
    'so hoa don': 'order_id', 'ma hd': 'order_id', 'so phieu ban': 'order_id',
    'thoi gian': 'date', 'ngay hd': 'date', 'ngay ban': 'date', 'ngay': 'date', 'ngaylap': 'date',
    'thoi gian tao': 'datetime', 'gio tao don': 'datetime', 'thoigianlap': 'datetime',
    'ma kh': 'cust_id', 'ma khach': 'cust_id', 'ma khach hang': 'cust_id', 'makhachhang': 'cust_id',
    'khach hang': 'cust_name', 'khach hang ten': 'cust_name', 'ten khach hang': 'cust_name',
    'tong tien hang': 'goods_total', 'tien hang truoc giam': 'goods_total', 'tongtienhang': 'goods_total',
    'giam gia': 'discount', 'tien giam': 'discount', 'chietkhau': 'discount',
    'tong sau giam gia': 'payable', 'tong thanh toan': 'payable', 'tongthanhtoan': 'payable',
    'khach da tra': 'paid', 'da tra': 'paid', 'khachdatra': 'paid',
    'cong no': 'debt', 'con no': 'debt', 'conphaithu': 'debt',
    'phi ship': 'ship_fee', 'phivanchuyen': 'ship_fee',
    'thuegtgt': 'vat',
    'phuong thuc thanh toan': 'pay_method', 'hinh thuc thanh toan': 'pay_method',
    'phuongthuctt': 'pay_method', 'pttt': 'pay_method', 'phuongthuc': 'pay_method',
    'trang thai': 'status', 'trang thai giao': 'status', 'tinhtrang': 'status',
    'kenh ban': 'channel', 'nguon': 'channel', 'kenhbanhang': 'channel',
    'nguoi ban': 'staff_id', 'ma nv ban': 'staff_id', 'manhanvien': 'staff_id', 'ma nv': 'staff_id',
    'ghi chu': 'note', 'diengiai': 'note',
    'sdt': 'phone', 'dien thoai': 'phone', 'sodienthoai': 'phone',
    # lines
    'stt': 'line_no', 'dong': 'line_no', 'sothutu': 'line_no',
    'ma hang': 'prod_id', 'barcode': 'prod_id', 'mahanghoa': 'prod_id', 'ma h hoa': 'prod_id',
    'ten hang': 'prod_name', 'hang hoa': 'prod_name', 'ten hang hoa': 'prod_name',
    'don vi tinh': 'unit', 'dvt': 'unit', 'dv': 'unit', 'donvitinh': 'unit',
    'so luong': 'qty', 's l': 'qty', 'sl': 'qty',
    'so luong quy doi': 'qty_base', 'sl dv co ban': 'qty_base', 'slquydoi': 'qty_base',
    'don gia': 'price', 'gia 1 dv': 'price', 'giadonvi': 'price',
    'thanh tien': 'amount', 'thanh tien d': 'amount', 'trigia': 'amount',
    'giam d': 'line_disc',
    'gia von': 'cost', 'gia von luc ban': 'cost', 'giavonlucban': 'cost',
    # catalogue
    'nhom hang': 'category', 'nhom hang 3 cap': 'category', 'danhmuc': 'category',
    'gia ban': 'cur_price', 'gia le': 'cur_price', 'giabanle': 'cur_price',
    'gia von hien tai': 'cost', 'giavonhang': 'cost',
    'dang kinh doanh': 'active', 'conkinhdoanh': 'active',
    'tengoi': 'name', 'donvicoso': 'base_unit',
    # payments
    'ma phieu': 'pay_id', 'ma thanh toan': 'pay_id', 'maphieuthu': 'pay_id',
    'gia tri': 'pay_amount', 'tien thu': 'pay_amount', 'so tien thu': 'pay_amount', 'sotienthu': 'pay_amount',
    'ngay tt': 'paid_at', 'ngaygiothu': 'paid_at',
    # purchases
    'ma nhap hang': 'rc_id', 'so phieu': 'rc_id', 'maphieunhap': 'rc_id',
    'ma ncc': 'supplier_id', 'manhacungcap': 'supplier_id',
    'can tra ncc': 'rc_total', 'tonggiatrinhap': 'rc_total',
    'ngaynhapkho': 'rc_date',
    'dongianhap': 'rc_price', 'gia nhap': 'rc_price',
    'nha cung cap': 'supplier',
    # stock
    'ton cuoi ky': 'stock', 'ton kho': 'stock', 'ton cuoi': 'stock',
    # units
    'quy doi': 'factor', 'he so': 'factor', 'bang': 'factor', 'hesoquydoi': 'factor',
    'quy cach': 'big_unit',
}


NAME_ALIASES.update({'phuong thuc tt': 'pay_method', 'da thu': 'paid', 'hinh thuc tt': 'pay_method'})
# keys go through the same normalisation as column names ('order id' -> 'orde id')
NAME_ALIASES = {norm(k): v for k, v in NAME_ALIASES.items()}


def map_desc(desc: str):
    d = norm(desc)
    for rx, field in DESC_RULES:
        if rx.search(d):
            return field
    return None


def map_columns(header, desc_map=None):
    """Return {field: column_index}. desc_map = {column: description}."""
    desc_n = {norm(k): v for k, v in (desc_map or {}).items()}
    out = {}
    for j, col in enumerate(header):
        if not col:
            continue
        cn = norm(col)
        field = None
        if cn in desc_n:
            field = map_desc(desc_n[cn])
        if field is None and not desc_n:
            field = NAME_ALIASES.get(cn)
        if field == 'EMPTY':
            out.setdefault('EMPTY_' + cn, j)
        elif field and field not in out:
            out[field] = j
    # a unit column next to a conversion factor is the big unit
    if 'factor' in out and 'unit' in out and 'big_unit' not in out:
        out['big_unit'] = out.pop('unit')
    # both a line discount and a plain discount: the plain one is order-level
    if 'line_disc' in out and 'discount' in out:
        out['order_disc'] = out.pop('discount')
    return out
