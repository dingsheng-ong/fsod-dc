#!/bin/sh

create_hashmap() {
    mktemp -d
}

get_hash() {
    echo "$1" | md5sum | awk '{print $1}'
}

add_item() {
    echo "${3}" > "${1}/$( get_hash $2 )"
}

get_item() {
    cat "${1}/$( get_hash $2 )"
}

cleanup() {
    rm -r "$1"
}
