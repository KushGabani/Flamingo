Dropzone.autoDiscover = false;
$(".dropzone").dropzone({
  addRemoveLinks: true,
  removedfile: function (file) {
    var name = file.name;

    $.ajax({
      type: "POST",
      url: "/deletefile",
      data: { name: name },
      sucess: function (data) {
        console.log("success: " + data);
      },
    });

    var _ref;
    return (_ref = file.previewElement) != null
      ? _ref.parentNode.removeChild(file.previewElement)
      : void 0;
  },
});
